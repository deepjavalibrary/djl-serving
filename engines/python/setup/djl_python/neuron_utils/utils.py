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

import torch
from typing import TYPE_CHECKING, Optional, Union, Dict
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


def parse_input_to_default_schema(input_map, tokenizer, config):
    input_keys = input_map.keys()
    if "model" in input_keys and "messages" in input_keys:
        return apply_chat_completion_template(input_map, tokenizer, config)


def parse_input_as_chat_completion(input_map, tokenizer, config):
    _inputs = input_map["inputs"]
    input_map["messages"] = _inputs
    if isinstance(_inputs, list) and not isinstance(_inputs[0], str):
        new_input_map = dict({"inputs": list()})
        if "parameters" in input_map.keys():
            new_input_map["parameters"] = input_map.get("parameters")
        intermediate_map = apply_chat_completion_template(
            input_map, tokenizer, config)
        new_input_map["inputs"] = intermediate_map.pop("inputs")
        return new_input_map
    else:
        return input_map


def apply_chat_completion_template(input_map: Dict, tokenizer, config):
    new_input_map = dict({"inputs": list(), "parameters": dict()})
    if hasattr(tokenizer, "apply_chat_template"):
        _messages = input_map.get("messages", input_map)
        chat_template_kwargs = dict(tokenize=False)
        if "add_generation_prompt" in input_map.keys():
            chat_template_kwargs["add_generation_prompt"] = True
        new_input_map["inputs"].append(
            tokenizer.apply_chat_template(_messages, **chat_template_kwargs))
    else:
        raise AttributeError(
            f"Cannot provide chat completion for tokenizer: {tokenizer.__class__}, "
            f"please ensure that your tokenizer supports chat templates.")
    if "temperature" in input_map:
        new_input_map["parameters"] = dict(
            new_input_map["parameters"],
            **{"temperature": input_map.get("temperature")})
    if "top_p" in input_map:
        new_input_map["parameters"] = dict(new_input_map["parameters"],
                                           **{"top_p": input_map.get("top_p")})
    if "logprobs" in input_map:
        new_input_map["parameters"] = dict(
            new_input_map["parameters"],
            **{"details": input_map.get("logprobs")})
    if "max_tokens" in input_map:
        new_input_map["parameters"] = dict(
            new_input_map["parameters"],
            **{"max_new_tokens": input_map.get("max_tokens")})
    else:
        new_input_map["parameters"] = dict(
            new_input_map["parameters"],
            **{"max_new_tokens": config.n_positions})
    return new_input_map


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
