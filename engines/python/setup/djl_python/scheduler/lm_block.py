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
from typing import List, Tuple

import torch


class LMBlock(ABC):

    @abstractmethod
    def __init__(self):
        """
        Set self.model to the input language model.
        """
        pass

    @abstractmethod
    def forward(self, inputs: List[torch.tensor], past_key_values: Tuple) -> Tuple[torch.tensor, Tuple, torch.tensor]:
        """
        Convert the variables between that used in the internal model's forward call and that used in the
        autoregressive search.

        Args:
            inputs (`List[torch.tensor]`):
                Contains [input_ids, position_ids, attention_mask], order preserved.
                `input_ids` and `position_ids` are of size (batch_size, input_seq_len),
                `attention_mask` is of size (batch_size, past_seq_len + input_seq_len).
            past_key_values (`Tuple`):
                The kv_cache. The required form of kv_cache used in the autoregressive search is Tuple[Tuple[key,
                value] * num_layers].
                key: (batch_size, num_heads, seq_len, kv_dim),
                value: (batch_size, num_heads, seq_len, kv_dim).
        Return:
            logits (`torch.tensor`):
                (batch_size, vocab_dim)
            past_key_values (`Tuple`):
                same as above.
            hidden_state ('torch.tensor`):
                (batch_size, seq_len, hidden_dim), the embedding of the tokens.
        """
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
            **self.kwargs)

        return logits, past_key_values, hidden_states[0]  # take the lowest hidden_states as token embedding


class BloomBlock(LMBlock):

    def __init__(self,
                 model,
                 use_cache=True,
                 output_attentions=False,
                 output_hidden_states=True,
                 **kwargs):
        super(BloomBlock, self).__init__()
        self.model = model
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.token_type_ids = None
        self.return_dict = False
        self.kwargs = kwargs

    def forward(self, inputs: List[torch.tensor], past_key_values):
        # kv: (batch, num_head, seq_len, kv_dim) <->
        # k: (batch*num_head, kv_dim, seq_len), v: (batch*num_head, seq_len, kv_dim)
        batch_size = inputs[0].shape[0]

        if past_key_values is not None:
            _, num_head, seq_len, kv_dim = past_key_values[0][0].shape
            new_kv_list = []
            for k, v in past_key_values:
                k_new = torch.permute(k.view(batch_size * num_head, seq_len, kv_dim), (0, 2, 1))
                v_new = v.view(batch_size * num_head, seq_len, kv_dim)
                new_kv_list.append((k_new, v_new))
            past_key_values = tuple(new_kv_list)

        logits, past_key_values, hidden_states = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
            **self.kwargs)

        _, kv_dim, seq_len = past_key_values[0][0].shape
        new_kv_list = []
        for k, v in past_key_values:
            k_new = torch.permute(k, (0, 2, 1)).view(batch_size, -1, seq_len, kv_dim)
            v_new = v.view(batch_size, -1, seq_len, kv_dim)
            new_kv_list.append((k_new, v_new))
        past_key_values = tuple(new_kv_list)

        return logits, past_key_values, hidden_states[0]  # take the lowest hidden_states as token embedding


