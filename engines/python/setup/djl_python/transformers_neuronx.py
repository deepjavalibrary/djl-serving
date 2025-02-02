#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import os
import copy
import logging
from transformers import AutoConfig, AutoTokenizer
from typing import Optional
from djl_python import Input, Output
from djl_python.encode_decode import encode
from djl_python.rolling_batch.neuron_rolling_batch import NeuronRollingBatch
from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
from djl_python.stable_diffusion_inf2 import StableDiffusionNeuronXService
from djl_python.streaming_utils import StreamingUtils
from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties, TnXGenerationStrategy, \
    TnXModelLoaders
from djl_python.properties_manager.properties import StreamingEnum
from djl_python.neuron_utils.model_loader import TNXModelLoader, OptimumModelLoader, TNXVllmModelLoader
from djl_python.neuron_utils.neuron_smart_default_utils import NeuronSmartDefaultUtils
from djl_python.neuron_utils.utils import task_from_config, build_vllm_rb_properties
from djl_python.utils import rolling_batch_inference, get_input_details
from djl_python.input_parser import parse_input_with_formatter

model = None

OPTIMUM_CAUSALLM_MODEL_TYPES = {"gpt2", "opt", "bloom", "llama", "mistral"}
OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS = {"llama", "mistral"}
VLLM_CONTINUOUS_BATCHING_MODELS = {"llama"}
NXDI_COMPILED_MODEL_FILE_NAME = "model.pt"


class TransformersNeuronXService(object):

    def __init__(self) -> None:
        """
        Initializes the TransformersNeuronXService class.

        This method initializes the instance variables of the class, including flags, model configurations,
        model loaders, tokenizers, rolling batch configurations, and input format configurations.

        Args:
            None

        Returns:
            None
        """
        self.initialized = False
        self.model = None
        self.model_config = None
        self.model_loader = None
        self.tokenizer = None
        self.rolling_batch = None
        self.config = None
        self.draft_model = None
        self.rolling_batch_config = dict()
        self.input_format_configs = None
        self._model_loader_class = TNXModelLoader
        self.input_format_args = None

    def optimum_not_supported(self) -> bool:
        """
        Checks if the model is not supported by Optimum.

        This function checks if the model architecture contains 'CausalLM' and if the model type is not in the list of
        supported Optimum models. It also checks if the model supports continuous batching and if rolling batch is
        disabled.

        Args:
            None

        Returns:
            bool: True if the model is not supported by Optimum, False otherwise.
        """
        support = False
        if self.model_config.architectures is not None and any(
                "CausalLM" in arch
                for arch in self.model_config.architectures):
            # Limit optimum model loading to implemented models listed in the constant above
            support = self.model_config.model_type not in OPTIMUM_CAUSALLM_MODEL_TYPES
            # Optimum only compiles for rolling batch for models that support it
            support = support or (
                self.model_config.model_type
                in OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS
                and self.config.rolling_batch == "disable")
        return support

    def vllm_not_supported(self) -> bool:
        """
        Checks if vLLM is not supported based on the model type and rolling batch configuration.

        Args:
            None

        Returns:
            bool: True if vLLM is not supported, False otherwise.
        """
        # Current support on vLLM is only continuous batching llama models
        if self.model_config.model_type not in VLLM_CONTINUOUS_BATCHING_MODELS and self.config.rolling_batch == "vllm":
            return True
        return False

    def set_model_loader_class(self) -> None:
        """
        Sets the model loader class based on the configuration.

        Args:
            None

        Returns:
            None
        """
        if self.config.model_loader == "tnx":
            if self.config.speculative_draft_model and self.config.rolling_batch == "vllm":
                self._model_loader_class = TNXVllmModelLoader
                logging.info(
                    "Loading model using TNXVllmModelLoader for speculative decoding..."
                )
            else:
                self._model_loader_class = TNXModelLoader
                logging.info("Loading model using TNXModelLoader...")
            return

        if self.config.model_loader == "optimum":
            if self.optimum_not_supported():
                raise AttributeError(
                    f"OptimumModelLoader does not support this config: {self.config}"
                )
            self._model_loader_class = OptimumModelLoader
            logging.info("Loading model using OptimumModelLoader...")
            return

        if self.config.model_loader == "nxdi":
            os.environ[
                'VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
            djl_neuron_compiled_artifacts_path = os.path.join(
                os.getenv("DJL_CACHE_DIR", "/tmp/.djl.ai"),
                "neuron-compiled-artifacts")
            nxdi_compiled_model_path = os.path.join(
                self.config.model_id_or_path, NXDI_COMPILED_MODEL_FILE_NAME)
            if self.config.save_mp_checkpoint_path:
                # If the compilation path is given by the user
                os.environ[
                    "NEURON_COMPILED_ARTIFACTS"] = self.config.save_mp_checkpoint_path
            elif os.path.isfile(nxdi_compiled_model_path):
                # if the compilation path already exists
                os.environ[
                    "NEURON_COMPILED_ARTIFACTS"] = self.config.model_id_or_path
            else:
                os.environ[
                    "NEURON_COMPILED_ARTIFACTS"] = djl_neuron_compiled_artifacts_path
            return

        if self.config.model_loader == "vllm":
            """vLLM does not need to set a model loader and instead defers model loading to the vLLM package"""
            if self.vllm_not_supported():
                raise AttributeError(
                    f"VllmModelLoader does not support this config: {self.config}"
                )

    def set_configs(self,
                    properties: dict,
                    is_partition: bool = False) -> None:
        """
        Sets the model configuration properties and performs necessary setup for model loading.

        Args:
            properties (dict): A dictionary containing model configuration properties.
            is_partition (bool): indicates whether we are saving pre-sharded checkpoints or not.
                                 We set some smart defaults for it.

        Returns:
            None
        """
        self.model_config = AutoConfig.from_pretrained(
            properties.get("model_id") or properties.get("model_dir"),
            revision=properties.get("revision"),
            trust_remote_code=properties.get("trust_remote_code"))

        utils = NeuronSmartDefaultUtils()
        utils.apply_smart_defaults(properties,
                                   copy.deepcopy(self.model_config.__dict__),
                                   is_partition=is_partition)

        self.config = TransformerNeuronXProperties(**properties)
        if self.config.rolling_batch != "disable":
            """batch_size needs to match max_rolling_batch_size for precompiled neuron models running rolling batch"""
            self.config.batch_size = self.config.max_rolling_batch_size

        if self.config.rolling_batch != "disable" and self.config.rolling_batch_strategy is None:
            if (self.model_config.model_type
                    in OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS
                    and self.config.max_rolling_batch_size > 1):
                self.config.rolling_batch_strategy = TnXGenerationStrategy.continuous_batching
            else:
                self.config.rolling_batch_strategy = TnXGenerationStrategy.naive_rolling_batch

        logging.info(f"Model loading properties: {self.config}")
        self.set_model_loader_class()
        logging.debug(f"Model loader class {self._model_loader_class}")
        if not self.config.task:
            self.config.task = task_from_config(self.model_config)

    def set_tokenizer(self) -> None:
        """
        Sets the tokenizer for the model based on the provided configuration.

        Args:
            None

        Returns:
            None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id_or_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            padding_side="left")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_rolling_batch(self, properties: dict) -> None:
        """
        Sets the rolling batch configuration for the model based on the provided properties.

        Args:
            properties (dict): A dictionary containing rolling batch configuration properties.

        Returns:
            None
        """
        if self.config.rolling_batch == "vllm":
            self.rolling_batch_config = build_vllm_rb_properties(properties)
            if self.model:
                self.rolling_batch_config["preloaded_model"] = self.model
            self.rolling_batch = VLLMRollingBatch(self.config.model_id_or_path,
                                                  self.rolling_batch_config)
        elif self.config.rolling_batch != "disable":
            if self.draft_model:
                self.rolling_batch_config["draft_model"] = self.draft_model
                self.rolling_batch_config[
                    "spec_length"] = self.config.speculative_length
            self.rolling_batch = NeuronRollingBatch(
                self.model, self.tokenizer, self.config.rolling_batch_strategy,
                self.config, **self.rolling_batch_config)

    def set_model_loader(self) -> None:
        """
        Sets the model loader instance based on the provided configuration and model configuration.

        Args:
            None

        Returns:
            None
        """
        self.model_loader = self._model_loader_class(
            config=self.config, model_config=self.model_config)

    @staticmethod
    def set_draft_model_properties(properties: dict) -> dict:
        """
        Sets the draft model properties for the provided model configuration.

        Args:
            properties (dict): A dictionary containing model configuration properties.

        Returns:
            dict: The updated model configuration properties with draft model settings.
        """
        draft_properties = copy.deepcopy(properties)
        # Optimum currently doesn't support speculative decoding
        draft_properties["model_loader"] = TnXModelLoaders.tnx
        draft_model_id = draft_properties.pop("speculative_draft_model")
        draft_properties["model_id"] = draft_model_id

        draft_tp_degree = draft_properties.pop("draft_model_tp_size", None)
        if draft_tp_degree:
            draft_properties["tensor_parallel_degree"] = draft_tp_degree

        draft_compiled = draft_properties.pop("draft_model_compiled_path",
                                              None)
        if draft_compiled:
            draft_properties["compiled_graph_path"] = draft_compiled
        return draft_properties

    def pre_model_load(self, properties: dict) -> None:
        """
        Prepares the model for loading by checking if a speculative draft model is specified in the properties.

        Args:
            properties (dict): A dictionary containing model configuration properties.

        Returns:
            None
        """
        rolling_batch = properties.get("rolling_batch", None)
        is_vllm = rolling_batch and rolling_batch == "vllm"
        if properties.get("speculative_draft_model") and not is_vllm:
            logging.info(
                f"Loading draft model {properties.get('speculative_draft_model')} ..."
            )
            draft_properties = self.set_draft_model_properties(properties)
            self.initialize_draft_model(draft_properties)
            logging.info(
                f"Loading target model {properties.get('model_id')} ...")

    def load_model(self) -> None:
        """
        Load the model based on the rolling batch and model loader configuration.

        This function checks the rolling batch and model loader configuration to determine
        how to load the model. If the rolling batch is set to "vllm" and the model loader
        is set to "vllm", the function returns without loading the model. If the rolling
        batch is set to "vllm", the function loads the unwrapped model using the model
        loader. Otherwise, the function loads the model using the model loader.

        Parameters:
            None

        Returns:
            None
        """
        if self.config.rolling_batch == "vllm" and (
                self.config.model_loader == "vllm"
                or self.config.model_loader == "nxdi"):
            """Model loading is being deferred to vLLMs model loader"""
            return
        elif self.config.rolling_batch == "vllm" and self.config.model_loader == "tnx":
            if self.config.speculative_draft_model:
                self.model = self.model_loader.load_model()
            else:
                raise ValueError(
                    f"Preloaded tnx model is only supported for speculative decoding for vllm."
                    f"Use vllm model_loader instead.")
        elif self.config.rolling_batch == "vllm":
            self.model = self.model_loader.load_unwrapped_model()
        else:
            self.model = self.model_loader.load_model()

    def initialize_draft_model(self, properties: dict) -> None:
        """
        Initializes a draft model based on the provided properties.

        Args:
            properties (dict): A dictionary containing model configuration properties.

        Returns:
            None
        """
        self.set_configs(properties)
        self.set_model_loader()
        self.draft_model = self.model_loader.load_unwrapped_model()

    def get_input_format_args(self) -> dict:
        """
        Returns a dictionary containing the input format arguments.

        The dictionary includes the following keys:
            - configs: The model configuration.
            - tokenizer: The tokenizer used by the model.
            - model_config: The model's configuration.
            - rolling_batch: The rolling batch configuration.

        Returns:
            dict: A dictionary containing the input format arguments.
        """
        return {
            "configs": self.config,
            "tokenizer": self.tokenizer,
            "model_config": self.model_config,
            "rolling_batch": self.rolling_batch
        }

    def initialize(self, properties: dict) -> None:
        """
        Initializes the object with the given properties.

        Args:
            properties (dict): A dictionary containing the properties to initialize the object.

        Returns:
            None
        """
        self.pre_model_load(properties)
        self.set_configs(properties)
        self.set_tokenizer()
        self.set_model_loader()
        self.load_model()
        self.set_rolling_batch(properties)
        self.input_format_args = self.get_input_format_args()
        self.initialized = True

    def partition(self, properties: dict) -> None:
        """
        Partitions the model based on the given properties.

        Args:
            properties (dict): A dictionary containing model configuration properties.

        Returns:
            None
        """
        self.pre_model_load(properties)
        self.set_configs(properties, is_partition=True)
        self.set_tokenizer()
        self.set_model_loader()
        self.model = self.model_loader.partition(
            self.config.save_mp_checkpoint_path,
            tokenizer=self.tokenizer,
            model_schema=self.config.partition_schema)
        self.set_rolling_batch(properties)
        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        """
        Performs inference on the given inputs using the model.

        Args:
            inputs (Input): The input data to perform inference on.

        Returns:
            Output: The output of the inference operation.
        """
        parsed_input = parse_input_with_formatter(inputs,
                                                  **self.input_format_args)
        errors = parsed_input.errors
        requests = parsed_input.requests
        outputs = Output()

        if self.rolling_batch:
            return rolling_batch_inference(parsed_input, inputs, outputs,
                                           self.rolling_batch)

        batch = parsed_input.batch
        input_data, input_size, parameters, _ = get_input_details(
            requests, errors, batch)
        model_kwargs = {}

        prompt_size = len(input_data)
        if prompt_size > self.config.batch_size:
            raise ValueError(
                f"Batch size {prompt_size} beyond the max_batch size the model can support {self.config.batch_size}"
            )

        for i in range(prompt_size, self.config.batch_size):
            input_data.append(self.tokenizer.eos_token)

        # clean KV cache
        self.model.reset_generation()
        if self.config.enable_streaming != StreamingEnum.false:
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.config.enable_streaming == StreamingEnum.huggingface:
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data, None,
                        **model_kwargs))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "transformers-neuronx")
                model_kwargs["engine"] = "transformers-neuronx"
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     "cpu", **model_kwargs))
            return outputs

        encoded_inputs = self.tokenizer.batch_encode_plus(input_data,
                                                          return_tensors="pt",
                                                          padding=True)
        use_sample = parameters.pop("use_sample", False)
        if use_sample:
            max_len = parameters.pop("max_length", self.config.n_positions)
            sample_length = parameters.pop("max_new_tokens", max_len)
            output_tokens = self.model.neuron_sample(encoded_inputs.input_ids,
                                                     sample_length,
                                                     **parameters)
        else:
            output_tokens = self.model.generate(
                input_ids=encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask,
                **parameters)
        prediction = self.tokenizer.batch_decode(output_tokens,
                                                 skip_special_tokens=True)

        # trim the input based on the actual size
        prediction = prediction[:prompt_size]
        prediction = [{"generated_text": s} for s in prediction]

        offset = 0
        for i, item in enumerate(batch):
            content_type = item.get_property("Content-Type")
            accept = item.get_property("Accept")
            if not accept:
                content_type = content_type if content_type else "application/json"
                accept = content_type if content_type.startswith(
                    "tensor/") else "application/json"
            elif "*/*" in accept:
                accept = "application/json"

            err = errors.get(i)
            if err:
                encode(outputs,
                       err,
                       accept,
                       key=inputs.get_content().key_at(i))
            else:
                encode(outputs,
                       prediction[offset:offset + input_size[i]],
                       accept,
                       key=inputs.get_content().key_at(i))
                offset += input_size[i]

        outputs.add_property("content-type", "application/json")

        return outputs


_service = TransformersNeuronXService()


def partition(inputs: Input) -> None:
    """
    Partitions the input data for the NeuronX service.

    Args:
        inputs (Input): The input data to be partitioned.

    Returns:
        None
    """
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionNeuronXService()
        _service.partition(inputs.get_properties())


def handle(inputs: Input) -> Optional[Output]:
    """
    Handles the input data for the NeuronX service.

    Args:
        inputs (Input): The input data to be handled.

    Returns:
        Optional[Output]: The output of the handled input data, or None if the input is empty.
    """
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionNeuronXService()
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warm up the model on startup
        return None

    return _service.inference(inputs)
