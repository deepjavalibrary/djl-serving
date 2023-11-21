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
import re
import json
import torch
from typing import TYPE_CHECKING, Optional, Union
from optimum.neuron import NeuronModelForCausalLM

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


def save_pretrained_split(model, save_directory):
    model.save_pretrained(save_directory,
                          save_function=save_split,
                          max_shard_size='10000GB',
                          safe_serialization=False)


_KEY_TO_FILENAME_JSON = 'key_to_filename.json'


def save_split(state_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    key_to_filename = {}
    for idx, key in enumerate(state_dict.keys()):
        key_to_filename[key] = f'p{idx}.{sanitize_file_name(key)}'
    with open(os.path.join(save_dir, _KEY_TO_FILENAME_JSON), 'w') as f:
        json.dump(key_to_filename, f, indent=2)
    for key, tensor in state_dict.items():
        torch.save(tensor, os.path.join(save_dir, key_to_filename[key]))


def sanitize_file_name(name):
    sanitized = name.strip().replace(' ', '_')
    sanitized = re.sub(r'(?u)[^-\w.]', '', sanitized)
    if sanitized in {'', '.', '..'}:
        raise ValueError(f'Could not sanitize "{name}" to file name.')
    return sanitized


class NeuronXModelAdapter(NeuronModelForCausalLM):

    def __init__(self,
                 model: torch.nn.Module,
                 config: "PretrainedConfig",
                 model_path: Union[str, "Path", "TemporaryDirectory"],
                 generation_config: Optional["GenerationConfig"] = None):
        super().__init__(model, config, model_path, generation_config)
        self.model_type = config.model_type
        self.sample_options = ["start_ids", "top_k"]
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
