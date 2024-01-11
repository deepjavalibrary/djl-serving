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
import time
import json
import shutil
import logging
import tempfile
import importlib
from typing import Optional
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
from transformers_neuronx.config import NeuronConfig, QuantizationConfig, ContinuousBatchingConfig
from transformers_neuronx.module import save_pretrained_split
from djl_python.neuron_utils.utils import NeuronXModelAdapter
from huggingface_hub import hf_hub_download

_neuronxcc_version: Optional[str] = None


class ModelLoader(ABC):

    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.model_config = kwargs.get("model_config", None)

    def init_load_path(self):
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        folder = f"inf2_{self.model_config.model_type}_{self.config.amp}"
        if not path:
            path = tempfile.gettempdir()
        folder_path = os.path.join(path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        return folder_path

    def get_load_path(self):
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
        "bloom": "bloom.model.BloomForSampling"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.load_path = None
        self.split_model_path = None
        self.compiled_graph_path = None
        self.neuron_config = None
        self.set_neuron_config()
        module_name, class_name = self.MODEL_TYPE_TO_CLS_LOADER[
            self.model_config.model_type].rsplit(".", maxsplit=1)
        module = importlib.import_module(f"transformers_neuronx.{module_name}")
        self._neuronx_class = getattr(module, class_name, None)
        if self._neuronx_class is None:
            raise ImportError(
                f"{class_name} not found in {module_name}. Please check transformers-neuronx version."
            )

    @staticmethod
    def get_neuronxcc_version():
        global _neuronxcc_version
        if _neuronxcc_version is not None:
            return _neuronxcc_version
        try:
            import neuronxcc
        except ImportError:
            raise ModuleNotFoundError(
                "NeuronX Compiler python package is not installed.")
        _neuronxcc_version = neuronxcc.__version__
        return _neuronxcc_version

    def set_neuron_config(self):
        neuron_config = {}
        if self.config.rolling_batch != "disable":
            neuron_config["continuous_batching"] = ContinuousBatchingConfig(
                batch_size_for_shared_caches=self.config.max_rolling_batch_size
            )

        if self.config.load_in_8bit:
            neuron_config["quant"] = QuantizationConfig(
                quant_dtype="s8", dequant_dtype=self.config.amp)
        self.neuron_config = NeuronConfig(**neuron_config)

    def get_model_specific_kwargs(self):
        model_kwargs = {
            "batch_size": self.config.batch_size,
            "amp": self.config.amp,
            "tp_degree": self.config.tensor_parallel_degree,
            "n_positions": self.config.n_positions,
        }
        if self.config.context_length_estimate is not None:
            model_kwargs[
                "context_length_estimate"] = self.config.context_length_estimate

        # Continuous batching requires positions and estimates as lists instead of int
        if self.config.rolling_batch != "disable":
            model_kwargs["n_positions"] = [self.config.n_positions]
            if self.config.context_length_estimate is None:
                model_kwargs["context_length_estimate"] = [
                    self.config.n_positions
                ]
            elif self.config.context_length_estimate != [
                    self.config.n_positions
            ]:
                raise RuntimeError(
                    f"context_length_estimate {self.config.context_length_estimate}"
                    f" need to be the same as n_positions {self.config.n_positions}"
                    f" You can also unset option.context_length_estimate to make continuous batching to work"
                )
        return model_kwargs

    def update_model_config_to_neuron(self):
        neuron_config = {
            "neuron": {
                "auto_cast_type": self.config.amp,
                "batch_size": self.config.batch_size,
                "compiler_type": "neuronx-cc",
                "compiler_version": self.get_neuronxcc_version(),
                "num_cores": self.config.tensor_parallel_degree,
                "sequence_length": self.config.n_positions,
                "task": "text-generation"
            }
        }
        self.model_config.update(neuron_config)

    def load_hf_model(self):
        logging.info(
            f"Start loading the model {self.config.model_id_or_path}...")
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_id_or_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            low_cpu_mem_usage=True)

    def load_inf2_model_from_disk(self):
        if not self.config.load_split_model:
            logging.info(f"Saving INF2 model to {self.load_path} ...")
            save_pretrained_split(self.model, self.load_path)
        model_kwargs = self.get_model_specific_kwargs()
        return self._neuronx_class.from_pretrained(
            self.load_path, neuron_config=self.neuron_config, **model_kwargs)

    def load_inf2_model_from_memory(self):
        model_kwargs = self.get_model_specific_kwargs()
        model = self._neuronx_class(self.model.config,
                                    neuron_config=self.neuron_config,
                                    **model_kwargs)
        model.load_state_dict_low_memory(self.model.state_dict())
        return model

    def set_load_path(self):
        if "neuron" in self.model_config.to_dict():
            self.config.load_split_model = True
            self.split_model_path = os.path.join(self.get_load_path(),
                                                 "checkpoint")
            self.compiled_graph_path = os.path.join(self.get_load_path(),
                                                    "compiled")
            self.load_path = self.split_model_path
        elif not self.config.load_split_model:
            self.model = self.load_hf_model()
            self.load_path = self.get_load_path()
        else:
            self.load_path = self.config.model_id_or_path

    def set_neuron_model(self):
        if self.config.low_cpu_mem_usage or self.config.load_split_model:
            logging.info("Transferring weights from HF to INF2 through disk")
            self.model = self.load_inf2_model_from_disk()
        else:
            logging.info("Transferring weights from HF to INF2 in-memory")
            self.model = self.load_inf2_model_from_memory()

    def compile_model(self):
        logging.info(f"LLM sharding and compiling Started ...")
        start = time.time()
        # TODO: workaround on Neuron Compiler bug for SM
        path = os.getcwd()
        os.chdir("/tmp")
        if self.compiled_graph_path:
            self.model.load(self.compiled_graph_path)
        self.model.to_neuron()
        os.chdir(path)
        elapsed = time.time() - start
        logging.info(
            f"SysHealth: LLM sharding and compilation latency: {elapsed} secs")

    def load_model(self, **kwargs):
        self.set_load_path()
        self.set_neuron_model()
        self.compile_model()
        self.update_model_config_to_neuron()
        self.model = NeuronXModelAdapter(self.model, self.model_config,
                                         self.load_path)
        return self.model

    def partition(self, save_path, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if self.config.load_split_model:
            raise ValueError(
                "Model partitioning does not support split model artifacts. Use normal model artifacts and rerun."
            )
        logging.info(f"Saving INF2 model to {save_path} ...")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)

        self.split_model_path = os.path.join(save_path, "checkpoint")
        self.compiled_graph_path = os.path.join(save_path, "compiled")
        os.mkdir(os.path.join(save_path, "compiled"))

        self.update_model_config_to_neuron()
        self.model_config.save_pretrained(save_path)
        self.model = self.load_hf_model()
        self.load_path = self.get_load_path()
        save_pretrained_split(self.model, self.load_path)
        self.model = self.load_inf2_model_from_disk()
        shutil.copytree(self.load_path, self.split_model_path)

        # Neuron compiler serialization workaround
        path = os.getcwd()
        os.chdir("/tmp")
        self.model.to_neuron()
        self.model.save(self.compiled_graph_path)
        os.chdir(path)

        if tokenizer:
            tokenizer.save_pretrained(save_path)

        self.model = NeuronXModelAdapter(self.model, self.model_config,
                                         self.load_path)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.compile_model = True
        self.compiler_args = kwargs.get("compiler_args", dict())
        if "neuron" in self.model_config.to_dict():
            self._validate_neuron_config()
            self.compile_model = False
        class_name = self.TASK_TO_MODEL_LOADER[self.config.task]
        if class_name == "NeuronModelForCausalLM":
            """Apply model adapter to expose the neuron model sample method"""
            self._optimum_class = NeuronXModelAdapter
        else:
            module = importlib.import_module(f"optimum.neuron")
            self._optimum_class = getattr(module, class_name, None)
            if self._optimum_class is None:
                raise ImportError(
                    f"{class_name} not found in optimum.neuron. Please check optimum neuron version."
                )

    def get_compiler_args(self):
        if self.compile_model:
            self.compiler_args["export"] = True
            self.compiler_args[
                "num_cores"] = self.config.tensor_parallel_degree
            self.compiler_args["auto_cast_type"] = self.config.amp
        return self.compiler_args

    def get_model_args(self):
        input_shapes = dict()
        if self.config.task in self.TASK_TO_MODEL_LOADER:
            input_shapes["batch_size"] = self.config.batch_size
            input_shapes["sequence_length"] = self.config.n_positions
        return input_shapes

    def load_optimum_model(self, compiler_args, input_shapes):
        logging.info(
            f"Start loading the model {self.config.model_id_or_path}...")
        return self._optimum_class.from_pretrained(
            self.config.model_id_or_path, **compiler_args, **input_shapes)

    def load_model(self):
        compiler_args = self.get_compiler_args()
        input_shapes = self.get_model_args()
        self.model = self.load_optimum_model(compiler_args=compiler_args,
                                             input_shapes=input_shapes)
        return self.model

    def partition(self, save_path, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        self.model = self.load_model()
        self.model.save_pretrained(save_path)
        if tokenizer:
            tokenizer.save_pretrained(save_path)
        return self.model

    def _validate_neuron_config(self):
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

    def __init__(self, *args, **kwargs):
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

    def get_model_class(self):
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

    def get_compiler_args(self):
        if self.compile_model:
            self.compiler_args["export"] = True
            self.compiler_args["auto_cast"] = "matmul"
            self.compiler_args["auto_cast_type"] = self.config.amp
        return self.compiler_args

    def get_model_args(self):
        input_shapes = dict()
        input_shapes["batch_size"] = int(self.config.batch_size)
        input_shapes["height"] = int(self.config.height)
        input_shapes["width"] = int(self.config.width)
        input_shapes["num_images_per_prompt"] = int(
            self.config.num_images_per_prompt)
        return input_shapes

    def load_optimum_pipeline(self, compiler_args, input_shapes, **kwargs):
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
        compiler_args = self.get_compiler_args()
        input_shapes = self.get_model_args()
        self.pipeline = self.load_optimum_pipeline(compiler_args=compiler_args,
                                                   input_shapes=input_shapes,
                                                   **kwargs)
        return self.pipeline

    def load_model(self, **kwargs):
        self.load_pipeline(**kwargs)
        return self.pipeline

    def save_pipeline(self, save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        self.pipeline.save_pretrained(save_path)

    def partition(self, save_path, **kwargs):
        self.load_pipeline(**kwargs)
        self.save_pipeline(save_path)
        return self.pipeline
