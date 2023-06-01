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
from __future__ import annotations

import torch
from djl_python.scheduler.utils import merge_tensors, trim_tensor, nudge_tensor
from typing import List, Tuple


class Batch:

    def __init__(self,
                 logits: torch.Tensor = None,
                 past_key_values=None):
        self.past_key_values = past_key_values
        self.logits = logits

    # merges another batch with itself.
    def merge(self, batch: Batch, seq_delta: int) -> Batch:
        logits = merge_tensors(self.logits,
                               batch.logits,
                               seq_delta=seq_delta,
                               seq_order=-1)

        past_key_values = []
        for kv_pair1, kv_pair2 in zip(self.past_key_values,
                                      batch.past_key_values):
            kv = tuple()
            for kv1, kv2 in zip(kv_pair1, kv_pair2):
                kv += (merge_tensors(kv1,
                                     kv2,
                                     seq_delta=seq_delta,
                                     seq_order=2), )
            past_key_values.append(kv)
        past_key_values = tuple(past_key_values)

        return Batch(past_key_values=past_key_values, logits=logits)

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):

        self.logits = trim_tensor(self.logits,
                                  keep_indices=keep_indices,
                                  trim_seq_len=trim_seq_len,
                                  seq_order=-1)

        past_key_values = []
        for k, v in self.past_key_values:
            k = trim_tensor(k,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=2)
            v = trim_tensor(v,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=2)
            past_key_values.append((k, v))
        self.past_key_values = tuple(past_key_values)

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor,
                                        init_kv_cache_len: int):
        past_key_values = []
        for k, v in self.past_key_values:
            past_key_values.append((nudge_tensor(k,
                                                 offsets,
                                                 init_kv_cache_len,
                                                 seq_order=2),
                                    nudge_tensor(v,
                                                 offsets,
                                                 init_kv_cache_len,
                                                 seq_order=2)))
        self.past_key_values = tuple(past_key_values)


class ContrastiveBatch(Batch):

    def __init__(
            self,
            past_hidden_states: torch.Tensor = None,
            past_key_values: Tuple = None,
            logits: torch.Tensor = None):
        self.past_hidden_states = past_hidden_states
        super().__init__(past_key_values=past_key_values, logits=logits)

    # merges another batch with itself.
    def merge(self, batch: ContrastiveBatch, seq_delta: int) -> ContrastiveBatch:
        past_hidden_states = merge_tensors(self.past_hidden_states,
                                           batch.past_hidden_states,
                                           seq_delta=seq_delta,
                                           seq_order=1)

        batch = super().merge(batch, seq_delta)

        return ContrastiveBatch(past_key_values=batch.past_key_values, logits=batch.logits,
                                past_hidden_states=past_hidden_states)

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):
        self.past_hidden_states = trim_tensor(self.past_hidden_states,
                                              keep_indices=keep_indices,
                                              trim_seq_len=trim_seq_len,
                                              seq_order=1)
        super().trim(keep_indices, trim_seq_len)

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor, init_kv_cache_len: int):
        # The past_hidden_states doesn't have the sequence length of the prefix_kv_cache part.
        super().nudge_to_squeeze_bubble_padding(offsets, init_kv_cache_len)
