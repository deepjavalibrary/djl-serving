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
import os
import time
import json
import shutil
import logging
import tempfile
import importlib
import re
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers_neuronx import NeuronAutoModelForCausalLM
from transformers_neuronx.config import NeuronConfig, QuantizationConfig, ContinuousBatchingConfig, \
    GenerationConfig as NeuronGenerationConfig
from djl_python.properties_manager.tnx_properties import TnXGenerationStrategy, TnXModelSchema, TnXMemoryLayout
from transformers_neuronx.module import save_pretrained_split
from djl_python.neuron_utils.utils import NeuronXRollingBatchModelAdapter, NeuronXDynamicBatchModelAdapter, \
    get_neuronxcc_version, build_context_length_estimates
from huggingface_hub import hf_hub_download

# Temporary Fix: These loggers are disabled during vLLM import.
# Remove when fixed in vLLM
logging.getLogger("NEURON_CC_WRAPPER").disabled = False
logging.getLogger("NEURON_CACHE").disabled = False

NXDI_COMPILED_MODEL_FILE_NAME = "model.pt"


class ModelLoader(ABC):

    def __init__(self, *args, **kwargs) -> None:
        self.config = kwargs.get("config")
        self.model_config = kwargs.get("model_config", None)
        self.move_read_only_neuron_cache()

    def move_read_only_neuron_cache(self) -> None:
        """
        If the currently set Neuron cache directory is read-only,
        change the Neuron cache directory to the default: /var/tmp/neuron-compile-cache.
        Copies the graphs from the original location to the default location.

        This is a workaround for passing the Neuron cache with the model on SM.
        Enables reading a Neuron cache located in the read-only dir /opt/ml/model.
        """
        cache_dir = os.environ.get("NEURON_COMPILE_CACHE_URL")
        default_cache_dir = "/var/tmp/neuron-compile-cache"
        if cache_dir:
            if not re.search(r"^s3:\/\/([^/]+)\/([\w\W]+)", cache_dir):
                cache_dir = os.path.abspath(cache_dir)
                if os.access(cache_dir,
                             os.R_OK) and not os.access(cache_dir, os.W_OK):
                    logging.info(
                        f"Neuron cache directory is set to an unwriteable location: {cache_dir}"
                    )
                    start = time.perf_counter()
                    shutil.copytree(cache_dir, default_cache_dir)
                    os.environ["NEURON_COMPILE_CACHE_URL"] = default_cache_dir
                    duration = time.perf_counter() - start
                    logging.info(
                        f"Copied neuron cache to the default location: {default_cache_dir}. "
                        "Using this directory as the Neuron cache."
                        f"\nCopying took: {duration} seconds")

    def init_load_path(self) -> str:
        """
        Gets the path where artifacts should be put.

        :return: path
        """
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        folder = f"inf2_{self.model_config.model_type}_{self.config.amp}"
        if not path:
            path = tempfile.gettempdir()
        folder_path = os.path.join(path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        return folder_path

    def get_load_path(self) -> str:
        """
        Gets the path where artifacts should be put, either from the config or using
        a default name based on the model name.

        :return: path
        """
        model_path = os.path.join(os.getcwd(), self.config.model_id_or_path)
        if os.path.isdir(model_path):
            load_path = model_path
        else:
            load_path = self.init_load_path()
        return load_path

    @abstractmethod
    def load_model(self, **kwargs):
        """Model loading method which returns a loaded model"""

    @abstractmethod
    def partition(self, save_path, **kwargs):
        """Model compilation and export method which returns a loaded model"""


class TNXModelLoader(ModelLoader):
    """Model loader and compiler for legacy compiled or HuggingFace precompiled artifacts"""
    MODEL_TYPE_TO_CLS_LOADER = {
        "opt": "opt.model.OPTForSampling",
        "gpt2": "gpt2.model.GPT2ForSampling",
        "gptj": "gptj.model.GPTJForSampling",
        "gpt_neox": "gptneox.model.GPTNeoXForSampling",
        "llama": "llama.model.LlamaForSampling",
        "mistral": "mistral.model.MistralForSampling",
        "mixtral": "mixtral.model.MixtralForSampling",
        "bloom": "bloom.model.BloomForSampling",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.load_path = None
        self.compiled_graph_path = None
        self.neuron_config = None
        self.generation_config = None
        self.use_continuous_batching = self.can_use_continuous_batching()
        self.set_neuron_config()

        # Assume safetensors until model download
        self.safetensors_format = True

        # Workaround for issue with NeuronAuto gpt model class, and split model loading
        self._neuronx_class = self.set_neuronx_class()
        self._adapter_class = self.set_adapter_class()

    def set_neuronx_class(self):
        try:
            module_name, class_name = self.MODEL_TYPE_TO_CLS_LOADER[
                self.model_config.model_type].rsplit(".", maxsplit=1)
        except KeyError as exc:
            raise KeyError(
                f"Unsupported model: {str(exc)}. Supported architectures: {list(self.MODEL_TYPE_TO_CLS_LOADER.keys())}"
            )
        module = importlib.import_module(f"transformers_neuronx.{module_name}")
        neuronx_class = getattr(module, class_name, None)
        if neuronx_class is None:
            raise ImportError(
                f"{class_name} not found in {module_name}. Please check transformers-neuronx version."
            )
        return neuronx_class

    def set_adapter_class(self):
        if self.config.rolling_batch != "disable":
            return NeuronXRollingBatchModelAdapter
        return NeuronXDynamicBatchModelAdapter

    def can_use_continuous_batching(self) -> bool:
        """
        Set configuration for continuous batching, currently all vllm implementations are continuous batching
        and batch size greater than 1 for tnx and lmi-dist support rolling batch.

        :return: bool indicating if continuous batching can be used
        """
        use_continuous_batching = (
            self.config.rolling_batch != "disable"
            and self.config.rolling_batch_strategy
            == TnXGenerationStrategy.continuous_batching)
        return use_continuous_batching

    def set_neuron_config(self) -> None:
        """
        Creates neuron config based on whether rolling batch, quantization, GQA is on
        """
        neuron_config = {}
        if self.use_continuous_batching:
            neuron_config["continuous_batching"] = ContinuousBatchingConfig(
                batch_size_for_shared_caches=self.config.max_rolling_batch_size
            )
        if self.config.load_in_8bit:
            neuron_config["quant"] = QuantizationConfig(
                quant_dtype="s8", dequant_dtype=self.config.amp)
        if self.config.group_query_attention is not None:
            neuron_config[
                "group_query_attention"] = self.config.group_query_attention
        if self.config.shard_over_sequence:
            neuron_config[
                "shard_over_sequence"] = self.config.shard_over_sequence
        if self.config.fuse_qkv:
            neuron_config["fuse_qkv"] = self.config.fuse_qkv
        if self.config.fuse_mlp:
            neuron_config["fuse_mlp"] = self.config.fuse_mlp
        if self.config.fused_rmsnorm_qkv:
            neuron_config["fused_rmsnorm_qkv"] = self.config.fused_rmsnorm_qkv
        if self.config.qkv_tiling:
            neuron_config["qkv_tiling"] = self.config.qkv_tiling
        if self.config.weight_tiling:
            neuron_config["weight_tiling"] = self.config.weight_tiling
        if self.config.collectives_layout:
            neuron_config[
                "collectives_layout"] = self.config.collectives_layout
        if self.config.attention_layout:
            neuron_config["attention_layout"] = self.config.attention_layout
        if self.config.cache_layout:
            neuron_config["cache_layout"] = self.config.cache_layout
        if self.config.all_reduce_dtype:
            neuron_config["all_reduce_dtype"] = self.config.all_reduce_dtype
        if self.config.cast_logits_dtype:
            neuron_config["cast_logits_dtype"] = self.config.cast_logits_dtype
        if self.config.on_device_embedding:
            neuron_config[
                "on_device_embedding"] = self.config.on_device_embedding
        if self.config.on_device_generation:
            if len(self.config.on_device_generation.keys()) > 0:
                neuron_config["on_device_generation"] = NeuronGenerationConfig(
                    **self.config.on_device_embedding_config)
        self.neuron_config = NeuronConfig(**neuron_config)

    def get_model_specific_kwargs(self) -> dict:
        """
        Populates a dictionary based on properties and detects incompatible options

        :return: Dictionary of properties
        """
        model_kwargs = {
            "batch_size": self.config.batch_size,
            "amp": self.config.amp,
            "tp_degree": self.config.tensor_parallel_degree,
            "n_positions": self.config.n_positions,
        }
        if self.config.revision is not None:
            model_kwargs["revision"] = self.config.revision
        if self.config.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage
        if self.config.trust_remote_code:
            model_kwargs["trust_remote_code"] = self.config.trust_remote_code
        if self.config.context_length_estimate is not None:
            model_kwargs[
                "context_length_estimate"] = self.config.context_length_estimate

        # Continuous batching requires positions and estimates as lists instead of int
        if self.use_continuous_batching:
            model_kwargs["n_positions"] = [self.config.n_positions]
            if self.config.context_length_estimate is None and self.config.cache_layout == TnXMemoryLayout.LAYOUT_BSH:
                model_kwargs["context_length_estimate"] = [
                    self.config.n_positions
                ]
            elif self.config.context_length_estimate is None:
                model_kwargs[
                    "context_length_estimate"] = build_context_length_estimates(
                        self.config.n_positions)
                logging.info(
                    f"Setting context_length_estimate to {model_kwargs['context_length_estimate']}"
                )
            elif self.config.context_length_estimate is not None and self.config.context_length_estimate[
                    0] != self.config.n_positions and self.config.cache_layout == TnXMemoryLayout.LAYOUT_BSH:
                raise RuntimeError(
                    f"context_length_estimate {self.config.context_length_estimate}"
                    f" need to be the same as n_positions {self.config.n_positions}"
                    f" when using alternative cache layouts,"
                    f" you can always unset cache_layout to support multi bucketing w/ continuous batching."
                )
        return model_kwargs

    def update_model_config_to_neuron(self) -> None:
        """
        Adds a key-value pair to the model config with neuron settings.
        """
        neuron_config = {
            "neuron": {
                "auto_cast_type": self.config.amp,
                "batch_size": self.config.batch_size,
                "checkpoint_id": None,
                "checkpoint_revision": None,
                "compiler_type": "neuronx-cc",
                "compiler_version": get_neuronxcc_version(),
                "num_cores": self.config.tensor_parallel_degree,
                "sequence_length": self.config.n_positions,
            },
        }
        self.model_config.update(neuron_config)

    def load_auto_model(self, model_path) -> "PreTrainedModel":
        logging.info(
            f"Start loading the model {self.config.model_id_or_path} using NeuronAutoModel..."
        )
        model_kwargs = self.get_model_specific_kwargs()
        return NeuronAutoModelForCausalLM.from_pretrained(
            model_path, neuron_config=self.neuron_config, **model_kwargs)

    def load_hf_model(self) -> "PreTrainedModel":
        logging.info(
            f"Start loading the model {self.config.model_id_or_path} using HuggingFace..."
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_id_or_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage)

    def load_inf2_model_from_disk(self) -> "PreTrainedModel":
        if not self.config.load_split_model:
            logging.info(
                f"Saving INF2 model to {self.load_path} as split model...")
            save_pretrained_split(self.model, self.load_path)
        model_kwargs = self.get_model_specific_kwargs()
        return self._neuronx_class.from_pretrained(
            self.load_path, neuron_config=self.neuron_config, **model_kwargs)

    @staticmethod
    def is_safetensors(path):
        return any(
            filename.endswith(".safetensors") for filename in os.listdir(path))

    def set_model_format(self):
        model_path = os.path.join(os.getcwd(), self.config.model_id_or_path)
        if os.path.isdir(model_path):
            self.safetensors_format = self.is_safetensors(model_path)

    def load_generation_config(self):
        if os.path.isfile(
                os.path.join(self.load_path, "generation_config.json")):
            self.generation_config = GenerationConfig.from_pretrained(
                self.load_path)
        elif os.path.isfile(
                os.path.join(self.config.model_id_or_path,
                             "generation_config.json")):
            self.generation_config = GenerationConfig.from_pretrained(
                self.load_path)
        else:
            try:
                self.generation_config = GenerationConfig.from_pretrained(
                    self.config.model_id_or_path)
            except OSError:
                logging.info(
                    "Unable to load generation config - defaulting to generation config from the models config.json"
                )

    def set_neuron_model(self) -> None:
        """
        Sets the path to which to load artifacts and loads the model - based on specified format
        """
        if "neuron" in self.model_config.to_dict(
        ) and not self.safetensors_format:
            self.compiled_graph_path = os.path.join(self.get_load_path(),
                                                    "compiled")
            if self.is_safetensors(self.compiled_graph_path):
                self.model = self.load_auto_model(self.compiled_graph_path)
                self.load_path = self.get_load_path()
            else:
                self.config.load_split_model = True
                self.load_path = self.compiled_graph_path
                self.model = self.load_inf2_model_from_disk()
        elif self.config.load_split_model:
            self.load_path = self.config.model_id_or_path
            self.model = self.load_inf2_model_from_disk()
        elif self.model_config.model_type == "gpt2":
            # Cannot use automodel and not a split model use case
            self.model = self.load_hf_model()
            self.load_path = self.get_load_path()
            self.model = self.load_inf2_model_from_disk()
        else:
            self.model = self.load_auto_model(self.config.model_id_or_path)
            self.load_path = self.get_load_path()

        if self.config.compiled_graph_path is not None and self.compiled_graph_path is None:
            self.compiled_graph_path = self.config.compiled_graph_path

    def maybe_compile_model(self) -> None:
        """
        Convert model to Neuron compiled format or load using compiled artifacts
        """
        logging.info(f"LLM sharding and compiling started...")
        start = time.time()
        # TODO: workaround on Neuron Compiler bug for SM
        path = os.getcwd()
        os.chdir("/tmp")
        if self.config.speculative_draft_model:
            logging.info(
                f"Enabling speculative decoding for {self.config.speculative_length} tokens..."
            )
            self.model.enable_speculative_decoder(
                self.config.speculative_length)
        if self.compiled_graph_path and os.path.isdir(
                self.compiled_graph_path):
            logging.info(
                f"Loading precompiled graph from {self.compiled_graph_path} ..."
            )
            self.model.load(self.compiled_graph_path)
        self.model.to_neuron()
        os.chdir(path)
        elapsed = time.time() - start
        logging.info(
            f"SysHealth: LLM sharding and compilation latency: {elapsed} secs")

    def compile_and_save(self, save_path) -> None:
        """
        Convert model to Neuron compiled format and save
        """
        logging.info(f"Saving preshareded model to {save_path}...")
        # Neuron compiler serialization workaround
        path = os.getcwd()
        os.chdir("/tmp")
        self.model.to_neuron()
        self.model.save(save_path, sharded_weights=True)

        with open(os.path.join(save_path, "VERSION"), "w+") as version_file:
            version_file.write(f"{get_neuronxcc_version()}")

        os.chdir(path)

    def load_model(self, **kwargs) -> "PretrainedModel":
        """
        Builds the NeuronX model.

        :return: model (NeuronXRollingBatchModelAdapter)
        """
        self.set_model_format()
        self.set_neuron_model()
        self.maybe_compile_model()
        self.update_model_config_to_neuron()
        self.load_generation_config()
        self.model = self._adapter_class(self.model, self.model_config,
                                         self.load_path,
                                         self.generation_config)
        return self.model

    def load_unwrapped_model(self) -> NeuronAutoModelForCausalLM:
        """
        Builds the NeuronX model.

        :return: model (NeuronAutoModelForCausalLM)
        """
        self.set_model_format()
        self.set_neuron_model()
        self.maybe_compile_model()
        self.update_model_config_to_neuron()
        return self.model

    def safetensors_partition(self, save_path: str):
        """
        Saves the model weights as safetensors, updates config to neuron, and adds compiled artifacts.

        :param save_path: Path to which to save the compiled model.
        """
        self.compiled_graph_path = os.path.join(save_path, "compiled")
        os.mkdir(self.compiled_graph_path)

        self.model = self.load_auto_model(self.config.model_id_or_path)
        self.compile_and_save(self.compiled_graph_path)

    def partition(self, save_path: str, **kwargs):
        """
        Builds the NeuronX model and additionally saves the compiled model.

        :param save_path: Path to which to save the compiled model.

        :return: model (NeuronXRollingBatchModelAdapter)
        """
        tokenizer = kwargs.get("tokenizer")
        model_schema = kwargs.get("model_schema")

        if self.config.load_split_model:
            raise ValueError(
                "Model partitioning does not support split model artifacts. Use normal model artifacts and rerun."
            )

        if not os.path.ismount(save_path):
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.mkdir(save_path)

        if model_schema == TnXModelSchema.compile_only:
            logging.info("Compiling model artifacts only...")
            self.model = self.load_auto_model(self.config.model_id_or_path)
            self.compile_and_save(save_path)
        else:
            logging.info("Compiling model to safetensors checkpoint ...")
            self.safetensors_partition(save_path)

        self.update_model_config_to_neuron()

        if model_schema != TnXModelSchema.compile_only:
            self.model_config.save_pretrained(save_path)
            if tokenizer:
                tokenizer.save_pretrained(save_path)

        self.model = self._adapter_class(self.model, self.model_config,
                                         save_path)
        return self.model


class OptimumModelLoader(ModelLoader):
    """Model loader and compiler for HuggingFace neuron model schema NLP artifacts"""
    TASK_TO_MODEL_LOADER = {
        "text-generation": "NeuronModelForCausalLM",
        "feature-extraction": "NeuronModelForFeatureExtraction",
        "fill-mask": "NeuronModelForMaskedLM",
        "question-answering": "NeuronModelForQuestionAnswering",
        "text-classification": "NeuronModelForSequenceClassification",
        "token-classification": "NeuronModelForTokenClassification"
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.load_path = None
        self.compile_model = True
        self.compiler_args = kwargs.get("compiler_args", dict())
        if "neuron" in self.model_config.to_dict():
            self._validate_neuron_config()
            self.compile_model = False
        class_name = self.TASK_TO_MODEL_LOADER[self.config.task]
        module = importlib.import_module(f"optimum.neuron")
        self._optimum_class = getattr(module, class_name, None)
        if self._optimum_class is None:
            raise ImportError(
                f"{class_name} not found in optimum.neuron. Please check optimum neuron version."
            )

    def get_compiler_args(self) -> dict:
        """
        Configures some options related to compiler args if applicable

        :return: compiler_args instance field
        """
        if self.compile_model:
            self.compiler_args["export"] = True
            self.compiler_args[
                "num_cores"] = self.config.tensor_parallel_degree
            self.compiler_args["auto_cast_type"] = self.config.amp
            self.compiler_args["task"] = self.config.task
        return self.compiler_args

    def get_model_args(self) -> dict:
        """
        Builds and returns a dict with properties relevant to the model

        :return: input_shapes, containing batch_size and sequence_length properties of the model
        """
        input_shapes = dict()
        if self.config.task in self.TASK_TO_MODEL_LOADER:
            input_shapes["batch_size"] = self.config.batch_size
            input_shapes["sequence_length"] = self.config.n_positions
        return input_shapes

    def set_generation_config(self):
        if hasattr(self.model.generation_config, "max_length"):
            self.model.generation_config.max_length = self.config.n_positions

    def load_optimum_model(
            self, compiler_args: dict,
            input_shapes: dict) -> NeuronXRollingBatchModelAdapter:
        """
        Helper function to load the model

        :param compiler_args: contains args needed for neuron compiler
        :param input_shapes: contains batch_size and sequence_length properties of the model

        :return: NeuronDecoderModel (generic Neuron model)
        """
        logging.info(
            f"Start loading the model {self.config.model_id_or_path}...")
        return self._optimum_class.from_pretrained(
            self.config.model_id_or_path, **compiler_args, **input_shapes)

    def load_model(self):
        """
        Builds the NeuronX model.

        :return: model (of type Union[NeuronBaseModel, NeuronDecoderModel])
        """
        compiler_args = self.get_compiler_args()
        input_shapes = self.get_model_args()
        self.model = self.load_optimum_model(compiler_args=compiler_args,
                                             input_shapes=input_shapes)
        if self.compile_model:
            self.model_config = self.model.config
        self.set_generation_config()
        self.model = NeuronXRollingBatchModelAdapter(
            self.model.model,
            self.model_config,
            self.get_load_path(),
            generation_config=self.model.generation_config)
        return self.model

    def partition(self, save_path: str, **kwargs):
        """
        Builds the NeuronX model and additionally saves the compiled model.

        :param save_path: Path to which to save the compiled model.

        :return: model (of type Union[NeuronBaseModel, NeuronDecoderModel])
        """
        logging.info("Partitioning model to optimum model schema")
        tokenizer = kwargs.get("tokenizer")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        compiler_args = self.get_compiler_args()
        input_shapes = self.get_model_args()
        self.model = self.load_optimum_model(compiler_args=compiler_args,
                                             input_shapes=input_shapes)
        if self.compile_model:
            self.model_config = self.model.config
        self.model.save_pretrained(save_path)
        if tokenizer:
            tokenizer.save_pretrained(save_path)
        self.model = NeuronXRollingBatchModelAdapter(
            self.model.model,
            self.model_config,
            self.get_load_path(),
            generation_config=self.model.generation_config)
        return self.model

    def _validate_neuron_config(self) -> None:
        """
        Detects if there are inconsistencies between the config of the model server
        and the model config that is put into the model artifacts.
        """
        errors = list()
        if self.config.batch_size != self.model_config.neuron["batch_size"]:
            errors.append(
                f"Model server configuration for batch size {self.config.batch_size} "
                f"does not match model config {self.model_config.neuron['batch_size']}"
            )
        if self.config.tensor_parallel_degree != self.model_config.neuron[
                "num_cores"]:
            errors.append(
                f"Model server configuration for tensor parallel degree {self.config.tensor_parallel_degree} "
                f"does not match model config {self.model_config.neuron['num_cores']}"
            )
        if self.config.n_positions != self.model_config.neuron[
                "sequence_length"]:
            errors.append(
                f"Model server configuration for n_positions {self.config.n_positions} "
                f"does not match model config {self.model_config.neuron['sequence_length']}"
            )
        if len(errors) > 0:
            raise ValueError(
                f"Mismatch between model server and compiled model configuration: {errors}"
            )


class OptimumStableDiffusionLoader(ModelLoader):
    """Pipeline loader and compiler for HuggingFace neuron model schema StableDiffusion artifacts"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline = None
        self.compile_model = True
        self.compiler_args = kwargs.get("compiler_args", dict())
        class_name = self.get_model_class()
        module = importlib.import_module(f"optimum.neuron")
        self._optimum_class = getattr(module, class_name, None)
        self.device_ids = None
        if self._optimum_class is None:
            raise ImportError(
                f"{class_name} not found in optimum.neuron. Please check optimum neuron version."
            )

    def get_model_class(self) -> str:
        """
        Gets the type of NeuronStableDiffusion class based on information inside model files

        :return: class name
        """
        model_path = os.path.join(os.getcwd(), self.config.model_id_or_path)
        if os.path.isdir(model_path):
            file_path = os.path.join(model_path, "model_index.json")
        else:
            file_path = hf_hub_download(repo_id=self.config.model_id_or_path,
                                        filename="model_index.json")
        with open(file_path) as file:
            model_index = json.load(file)
            pipeline_class_name = model_index.get("_class_name", "")
            if "Neuron" in pipeline_class_name:
                self.compile_model = False
            if "XL" in pipeline_class_name:
                return "NeuronStableDiffusionXLPipeline"
        return "NeuronStableDiffusionPipeline"

    def get_compiler_args(self) -> dict:
        """
        Configures some options related to compiler args if applicable

        :return: compiler_args instance field
        """
        if self.compile_model:
            self.compiler_args["export"] = True
            self.compiler_args["auto_cast"] = "matmul"
            self.compiler_args["auto_cast_type"] = self.config.amp
        return self.compiler_args

    def get_model_args(self) -> dict:
        """
        Builds and returns a dict with properties relevant to the model

        :return: input_shapes, containing batch_size and sequence_length properties of the model
        """
        input_shapes = dict()
        input_shapes["batch_size"] = int(self.config.batch_size)
        input_shapes["height"] = int(self.config.height)
        input_shapes["width"] = int(self.config.width)
        input_shapes["num_images_per_prompt"] = int(
            self.config.num_images_per_prompt)
        return input_shapes

    def load_optimum_pipeline(self, compiler_args: dict, input_shapes: dict,
                              **kwargs):
        """
        Helper function for loading Optimum pipeline

        :param compiler_args: Contains some options
        :param input_shapes: contains some input properties of the model

        :return: Union[NeuronStableDiffusionPipeline, NeuronStableDiffusionXLPipeline]
        """
        logging.info(
            f"Start loading the model {self.config.model_id_or_path}...")
        return self._optimum_class.from_pretrained(
            self.config.model_id_or_path,
            device_ids=[
                i for i in range(int(self.config.tensor_parallel_degree))
            ],
            **compiler_args,
            **input_shapes,
            **kwargs)

    def load_pipeline(self, **kwargs):
        """
        Gets relevant parameters and builds the stable diffusion pipeline.

        :return: Union[NeuronStableDiffusionPipeline, NeuronStableDiffusionXLPipeline]
        """
        compiler_args = self.get_compiler_args()
        input_shapes = self.get_model_args()
        self.pipeline = self.load_optimum_pipeline(compiler_args=compiler_args,
                                                   input_shapes=input_shapes,
                                                   **kwargs)
        return self.pipeline

    def load_model(self, **kwargs):
        """
        A wrapper for load_pipeline() to observe the same method signatures as superclass
        """
        self.load_pipeline(**kwargs)
        return self.pipeline

    def save_pipeline(self, save_path: str):
        """
        Saves the model to a location provided by save_path

        :param save_path: path to which to save the model
        """
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        self.pipeline.save_pretrained(save_path)

    def partition(self, save_path: str, **kwargs):
        """
        Loads and saves model

        :param save_path: path to which to save the model
        """
        self.load_pipeline(**kwargs)
        self.save_pipeline(save_path)
        return self.pipeline


class TNXVllmModelLoader(ModelLoader):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generation_config = self.generate_generation_config()
        self.model_config.generation_config = self.generation_config
        self.neuron_config = self.generate_neuron_config()

    def generate_generation_config(self):
        generation_config = NeuronGenerationConfig(
            max_length=self.config.n_positions,
            do_sample=True,
            per_batch_line=True,
            top_k=[1] * self.config.max_rolling_batch_size,
            top_p=[1] * self.config.max_rolling_batch_size,
            temperature=[1] * self.config.max_rolling_batch_size,
            dynamic=True,
            global_top_k=256,
            deterministic=True,
        )
        return generation_config

    def generate_neuron_config(self):
        continuous_batching_config = None
        if self.config.max_rolling_batch_size > 1:
            continuous_batching_config = ContinuousBatchingConfig(
                batch_size_for_shared_caches=self.config.max_rolling_batch_size
            )

        on_dev_sampling_config = None
        if self.config.on_device_generation:
            on_dev_sampling_config = copy.deepcopy(
                self.model_config.generation_config)

        quant_config = None
        if self.config.neuron_quant:
            quant_dtype = 's8'
            dequant_dtype = 'bf16'
            logging.info(f'Quantization is enabled. '
                         f'quant_dtype: {quant_dtype}, '
                         f'dequant_dtype: {dequant_dtype}')
            quant_config = QuantizationConfig(quant_dtype=quant_dtype,
                                              dequant_dtype=dequant_dtype)

        def get_default_if_none(value, default):
            return default if value is None else value

        attention_layout = get_default_if_none(self.config.attention_layout,
                                               TnXMemoryLayout.LAYOUT_BSH)
        collectives_layout = get_default_if_none(
            self.config.collectives_layout, TnXMemoryLayout.LAYOUT_BSH)
        sequence_parallel_norm_threshold = get_default_if_none(
            self.config.sequence_parallel_norm_threshold, 1)
        sequence_parallel = get_default_if_none(self.config.sequence_parallel,
                                                True)

        weight_tiling = get_default_if_none(
            self.config.weight_tiling, False if self.config.multi_node
            or self.config.speculative_draft_model else True)
        qkv_tiling = get_default_if_none(
            self.config.qkv_tiling, False if self.config.multi_node
            or self.config.speculative_draft_model else True)

        fuse_mlp = get_default_if_none(
            self.config.fuse_mlp, True if self.config.multi_node else False)
        mlp_out_weight_transpose = get_default_if_none(
            self.config.mlp_out_weight_transpose,
            True if self.config.multi_node else False)
        fuse_qkv = get_default_if_none(self.config.fuse_qkv, True)

        neuron_config = NeuronConfig(
            continuous_batching=continuous_batching_config,
            attention_layout=attention_layout.value,
            on_device_embedding=self.config.on_device_embedding,
            on_device_generation=on_dev_sampling_config,
            collectives_layout=collectives_layout.value,
            quant=quant_config,
            fuse_qkv=fuse_qkv,
            compilation_worker_count=self.config.compilation_worker_count,
            sequence_parallel_norm=sequence_parallel,
            sequence_parallel_norm_threshold=sequence_parallel_norm_threshold,
            shard_over_sequence=self.config.shard_over_sequence,
            weight_tiling=weight_tiling,
            qkv_tiling=qkv_tiling,
            fuse_mlp=fuse_mlp,
            mlp_out_weight_transpose=mlp_out_weight_transpose,
        )
        logging.debug(f'Neuron config: {neuron_config}')
        return neuron_config

    def load_model(self, **kwargs):
        from vllm.model_executor.model_loader.neuron import get_neuron_sd_model
        if self.config.context_length_estimate is None:
            self.config.context_length_estimate = build_context_length_estimates(
                self.config.n_positions)

        self.model = get_neuron_sd_model(
            model_name=self.config.model_id_or_path,
            tensor_parallel_degree=self.config.tensor_parallel_degree,
            model_config=self.model_config,
            neuron_config=self.neuron_config,
            speculative_draft_model=self.config.speculative_draft_model,
            num_speculative_tokens=self.config.speculative_length,
            max_num_seqs=self.config.max_rolling_batch_size,
            max_model_len=self.config.n_positions,
            predefined_buckets=self.config.context_length_estimate,
            dtype=self.config.amp,
            neuron_cc_pipeline_factor=self.config.neuron_cc_pipeline_factor,
            download_dir=kwargs.get('download_dir'))
        return self.model

    def partition(self, save_path, **kwargs):
        kwargs['download_dir'] = save_path
        self.load_model(**kwargs)


class NxdiVllmModelLoader(ModelLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
        djl_neuron_compiled_artifacts_path = os.path.join(
            os.getenv("DJL_CACHE_DIR", "/tmp/.djl.ai"),
            "neuron-compiled-artifacts")
        nxdi_compiled_model_path = os.path.join(self.config.model_id_or_path,
                                                NXDI_COMPILED_MODEL_FILE_NAME)
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

    def load_model(self, **kwargs):
        # Offloading loading model to vLLM
        return None

    def partition(self, save_path, **kwargs):
        # Offloading partition of model to vLLM.
        return None
