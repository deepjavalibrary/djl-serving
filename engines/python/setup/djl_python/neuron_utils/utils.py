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
import torch
from typing import TYPE_CHECKING, Optional, Union
from djl_python.transformers_neuronx_scheduler.optimum_modeling import OptimumModelForCausalLM
from optimum.exporters.neuron.model_configs import *
from optimum.exporters.tasks import TasksManager

if TYPE_CHECKING:
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from transformers import GenerationConfig, PretrainedConfig

ARCHITECTURES_2_TASK = {
    "ForQuestionAnswering": "question-answering",
    "ForTokenClassification": "token-classification",
    "ForSequenceClassification": "text-classification",
    "ForMultipleChoice": "multiple-choice",
    "ForMaskedLM": "fill-mask",
    "ForCausalLM": "text-generation",
    "ForFeatureExtraction": "feature-extraction"
}

TNX_ONLY_PROPERTIES = [
    "quantize", "neuron_optimize_level", "enable_mixed_precision_accumulation",
    "enable_saturate_infinity", "unroll", "load_in_8bit", "low_cpu_mem_usage",
    "load_split_model", "context_length_estimate", "amp",
    "compiled_graph_path", "draft_model_compiled_path",
    "speculative_draft_model", "speculative_length", "draft_model_tp_size",
    "task", "save_mp_checkpoint_path", "group_query_attention", "model_loader",
    "rolling_batch_strategy", "fuse_qkv", "on_device_embedding_config",
    "attention_layout", "collectives_layout", "cache_layout",
    "partition_schema", "all_reduce_dtype", "cast_logits_dtype"
]

_neuronxcc_version: Optional[str] = None


def task_from_config(config) -> str:
    if config.architectures is None:
        return "text-generation"
    arch = config.to_dict()['architectures'][0]
    for key, value in ARCHITECTURES_2_TASK.items():
        if key in arch:
            return value
    return "text-generation"


def get_exporter(config, task):
    return TasksManager.get_exporter_config_constructor(
        model_type=config.model_type, exporter="neuron", task=task)()


def get_neuronxcc_version() -> str:
    """
    Gets version of NeuronX Compiler

    :return: NeuronX compiler version
    """
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


def build_vllm_rb_properties(properties: dict) -> dict:
    vllm_properties = copy.deepcopy(properties)
    vllm_properties["device"] = "neuron"
    vllm_properties["max_model_len"] = properties["n_positions"]
    del vllm_properties["n_positions"]

    for tnx_only_property in TNX_ONLY_PROPERTIES:
        if tnx_only_property in vllm_properties:
            del vllm_properties[tnx_only_property]

    return vllm_properties


class NeuronXModelAdapter(OptimumModelForCausalLM):

    def __init__(self,
                 model: torch.nn.Module,
                 config: "PretrainedConfig",
                 model_path: Union[str, "Path", "TemporaryDirectory"],
                 generation_config: Optional["GenerationConfig"] = None):
        super().__init__(model, config, model_path, generation_config)
        self.model_type = config.model_type
        self.sample_options = ["start_ids", "top_k"]
        self.cur_len = 0
        if self.model_type == "llama":
            self.sample_options = self.sample_options + [
                "top_p", "eos_token_override", "temperature", "streamer"
            ]

    def neuron_sample(self, *args, **kwargs):
        sample_kwargs = self.simple_sample_parser(**kwargs)
        return self.model.sample(*args, **sample_kwargs)

    def save(self, path):
        return self.model.save(path)

    def simple_sample_parser(self, **kwargs):
        parsed_kwargs = dict()
        for key in self.sample_options:
            if key in kwargs:
                parsed_kwargs[key] = kwargs[key]
        return parsed_kwargs
