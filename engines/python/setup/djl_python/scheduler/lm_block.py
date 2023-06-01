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
from typing import List

import torch


class LMBlock(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs, past_key_values):
        pass


class HuggingfaceBlock(LMBlock):

    def __init__(self,
                 model,
                 use_cache=True,
                 output_attentions=False,
                 output_hidden_states=True,
                 **kwargs):
        super(HuggingfaceBlock, self).__init__()

        self.model = model
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.token_type_ids = None
        self.return_dict = False
        self.kwargs = kwargs

    def forward(self, inputs: List[torch.tensor], past_key_values):
        logits, past_key_values, hidden_states = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
            token_type_ids=self.token_type_ids,
            **self.kwargs)

        return logits, past_key_values, hidden_states[0]  # take the lowest hidden_states as token embedding


