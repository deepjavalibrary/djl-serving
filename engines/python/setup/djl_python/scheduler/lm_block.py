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

from abc import ABC, abstractmethod
from transformers import GPT2LMHeadModel
from typing import List, Dict

import torch

class GPT_config:
    def __init__(self):
        self.numAttentionHeads = 12
        self.numLayers = 12
        self.hiddenStateDim = 768
        self.logitsDim = 50257
        self.kvDim = 64


class LMBlock(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, input, past_key_values):
        pass


class PtLMBlock(LMBlock):
    def __init__(self, model_urls: List[str], gpt_config: GPT_config):
        super(PtLMBlock, self).__init__()
        self.blocks = [torch.jit.load(url) for url in model_urls]
        self.gpt_config = gpt_config

    def forward(self, input: List[torch.tensor], past_key_values):
        return self.blocks[0](*input) if past_key_values is None else self.blocks[1](*input, past_key_values)


class OrtLMBlock(LMBlock):
    def __init__(self, model_urls: List[str], gpt_config: GPT_config):
        super(OrtLMBlock, self).__init__()
        self.blocks = [torch.jit.load(url) for url in model_urls]
        self.gpt_config = gpt_config

    def forward(self, input: List[torch.tensor], past_key_values):
        raise ("Not implemented yet")


class HuggingfaceGTP2Block(LMBlock):
    def __init__(self, model_urls: List[str], config: Dict):
        super(HuggingfaceGTP2Block, self).__init__()
        self.config = {'use_cache': config.get('use_cache', True),
                       'token_type_ids': config.get('token_type_ids', None),
                       'return_dict': config.get('return_dict', False),
                       'output_attentions': config.get('output_attentions', False),
                       'output_hidden_states': config.get('output_hidden_states', True)}
        model = GPT2LMHeadModel.from_pretrained(model_urls[0])  # it contains model.eval()
        self.blocks = [model]

    def forward(self, input: List[torch.tensor], past_key_values):
        return self.blocks[0].forward(input_ids=input[0],
                                      position_ids=input[1],
                                      attention_mask=input[2],
                                      past_key_values=past_key_values,
                                      **self.config)
