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

import copy
import logging
from transformers import AutoConfig, AutoTokenizer
from djl_python import Input, Output
from djl_python.encode_decode import encode
from djl_python.rolling_batch.neuron_rolling_batch import NeuronRollingBatch
from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
from djl_python.stable_diffusion_inf2 import StableDiffusionNeuronXService
from djl_python.streaming_utils import StreamingUtils
from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties, TnXGenerationStrategy, \
    TnXModelLoaders
from djl_python.properties_manager.properties import StreamingEnum, is_rolling_batch_enabled
from djl_python.neuron_utils.model_loader import TNXModelLoader, OptimumModelLoader
from djl_python.neuron_utils.utils import task_from_config, build_vllm_rb_properties
from djl_python.utils import InputFormatConfigs, parse_input_with_formatter
from typing import Tuple, List

model = None

OPTIMUM_CAUSALLM_MODEL_TYPES = {"gpt2", "opt", "bloom", "llama", "mistral"}
OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS = {"llama", "mistral"}
VLLM_CONTINUOUS_BATCHING_MODELS = {"llama"}


class TransformersNeuronXService(object):

    def __init__(self) -> None:
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

    def optimum_not_supported(self) -> bool:
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
        # Current support on vLLM is only continuous batching llama models
        if self.model_config.model_type not in VLLM_CONTINUOUS_BATCHING_MODELS and self.config.rolling_batch == "vllm":
            return True
        return False

    def set_model_loader_class(self):
        if self.config.model_loader == "tnx":
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

        if self.config.model_loader == "vllm":
            """vLLM does not need to set a model loader and instead defers model loading to the vLLM package"""
            if self.vllm_not_supported():
                raise AttributeError(
                    f"VllmModelLoader does not support this config: {self.config}"
                )

    def set_configs(self, properties: dict):
        self.config = TransformerNeuronXProperties(**properties)
        if self.config.rolling_batch != "disable":
            """batch_size needs to match max_rolling_batch_size for precompiled neuron models running rolling batch"""
            self.config.batch_size = self.config.max_rolling_batch_size

        self.model_config = AutoConfig.from_pretrained(
            self.config.model_id_or_path,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code)

        if self.config.rolling_batch != "disable" and self.config.rolling_batch_strategy is None:
            if (self.model_config.model_type
                    in OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS
                    and self.config.max_rolling_batch_size > 1):
                self.config.rolling_batch_strategy = TnXGenerationStrategy.continuous_batching
            else:
                self.config.rolling_batch_strategy = TnXGenerationStrategy.naive_rolling_batch

        logging.info(f"Model loading properties: {self.config}")
        self.set_model_loader_class()
        if not self.config.task:
            self.config.task = task_from_config(self.model_config)

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id_or_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            padding_side="left")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_rolling_batch(self, properties: dict):
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

    def set_model_loader(self):
        self.model_loader = self._model_loader_class(
            config=self.config, model_config=self.model_config)

    @staticmethod
    def set_draft_model_properties(properties: dict) -> dict:
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

    def pre_model_load(self, properties: dict):
        if properties.get("speculative_draft_model"):
            logging.info(
                f"Loading draft model {properties.get('speculative_draft_model')} ..."
            )
            draft_properties = self.set_draft_model_properties(properties)
            self.initialize_draft_model(draft_properties)
            logging.info(
                f"Loading target model {properties.get('model_id')} ...")

    def load_model(self):
        if self.config.rolling_batch == "vllm" and self.config.model_loader == "vllm":
            """Model loading is being deferred to vLLMs model loader"""
            return
        elif self.config.rolling_batch == "vllm":
            self.model = self.model_loader.load_unwrapped_model()
        else:
            self.model = self.model_loader.load_model()

    def initialize_draft_model(self, properties: dict):
        self.set_configs(properties)
        self.set_model_loader()
        self.draft_model = self.model_loader.load_unwrapped_model()

    def initialize(self, properties: dict):
        self.pre_model_load(properties)
        self.set_configs(properties)
        self.set_tokenizer()
        self.set_model_loader()
        self.load_model()
        self.set_rolling_batch(properties)
        self.input_format_configs = InputFormatConfigs(
            is_rolling_batch=is_rolling_batch_enabled(
                self.config.rolling_batch),
            is_adapters_supported=False,
            tokenizer=self.tokenizer,
            output_formatter=self.config.output_formatter)
        self.initialized = True

    def parse_input(
        self, inputs: Input, tokenizer, output_formatter
    ) -> Tuple[List[str], List[int], List[dict], dict, list]:
        """
        Preprocessing function that extracts information from Input objects.

        :param output_formatter: output formatter for the request
        :param inputs :(Input) a batch of inputs, each corresponding to a new request
        :param tokenizer: the tokenizer used for inference

        :return input_data (List[str]): a list of strings, each string being the prompt in a new request
        :return input_size (List[int]): a list of ints being the size of each new request
        :return parameters (List[dict]): parameters pertaining to each request
        :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
        :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
        """
        parsed_input = parse_input_with_formatter(
            inputs, input_format_configs=self.input_format_configs)
        return parsed_input.input_data, parsed_input.input_size, parsed_input.parameters, parsed_input.errors, parsed_input.batch

    def partition(self, properties: dict):
        self.pre_model_load(properties)
        self.set_configs(properties)
        self.set_tokenizer()
        self.set_model_loader()
        self.model = self.model_loader.partition(
            self.config.save_mp_checkpoint_path,
            tokenizer=self.tokenizer,
            model_schema=self.config.partition_schema)
        self.set_rolling_batch(properties)
        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs, self.tokenizer, self.config.output_formatter)
        outputs = Output()

        if self.rolling_batch:
            if inputs.get_property("reset_rollingbatch"):
                self.rolling_batch.reset()
            result = self.rolling_batch.inference(input_data, parameters)
            idx = 0
            for i in range(len(batch)):
                err = errors.get(i)
                if err:
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                    outputs.add_property(f"batch_{i}_Content-Type",
                                         "application/json")
                else:
                    content_type = result[idx].pop("content_type")
                    outputs.add(Output.binary_encode(result[idx]),
                                key="data",
                                batch_index=i)
                    if content_type is not None:
                        outputs.add_property(f"batch_{i}_Content-Type",
                                             content_type)
                    idx += 1

            return outputs

        parameters = parameters[0]
        # Remove rolling batch default parameters
        parameters.pop("output_formatter", None)
        parameters.pop("stream", None)
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


def partition(inputs: Input):
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionNeuronXService()
        _service.partition(inputs.get_properties())


def handle(inputs: Input):
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionNeuronXService()
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warm up the model on startup
        return None

    return _service.inference(inputs)
