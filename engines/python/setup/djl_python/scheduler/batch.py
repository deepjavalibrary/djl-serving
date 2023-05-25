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


class Batch:

    def __init__(self,
                 past_output_ids: torch.Tensor = None,
                 past_attention_mask: torch.Tensor = None,
                 logits: torch.Tensor = None,
                 past_key_values=None):
        self.past_output_ids = past_output_ids
        self.past_key_values = past_key_values
        self.past_attention_mask = past_attention_mask
        self.logits = logits

    # merges another batch with itself.
    def merge(self, batch: Batch, seq_delta: int) -> Batch:
        past_output_ids = merge_tensors(self.past_output_ids,
                                        batch.past_output_ids,
                                        seq_delta=seq_delta,
                                        seq_order=1,
                                        is_pad_token=True)
        past_attention_mask = merge_tensors(self.past_attention_mask,
                                            batch.past_attention_mask,
                                            seq_delta=seq_delta,
                                            seq_order=1)

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
                                     seq_order=2),)
            past_key_values.append(kv)
        past_key_values = tuple(past_key_values)

        return Batch(past_output_ids=past_output_ids,
                     past_attention_mask=past_attention_mask,
                     logits=logits,
                     past_key_values=past_key_values)

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):
        self.past_output_ids = trim_tensor(self.past_output_ids,
                                           keep_indices=keep_indices,
                                           trim_seq_len=trim_seq_len,
                                           seq_order=1)

        self.past_attention_mask = trim_tensor(self.past_attention_mask,
                                               keep_indices=keep_indices,
                                               trim_seq_len=trim_seq_len,
                                               seq_order=1)

        self.logits = trim_tensor(self.logits,
                                  keep_indices=keep_indices,
                                  trim_seq_len=trim_seq_len,
                                  seq_order=-1)

        past_key_values = []
        for k, v in self.past_key_values:
            k = trim_tensor(k, keep_indices=keep_indices, trim_seq_len=trim_seq_len, seq_order=2)
            v = trim_tensor(v, keep_indices=keep_indices, trim_seq_len=trim_seq_len, seq_order=2)
            past_key_values.append((k, v))
        self.past_key_values = tuple(past_key_values)

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor, init_kv_cache_len: int):

        self.past_attention_mask = nudge_tensor(self.past_attention_mask, offsets, init_kv_cache_len, seq_order=1)

        past_key_values = []
        for k, v in self.past_key_values:
            past_key_values.append((nudge_tensor(k, offsets, init_kv_cache_len, seq_order=2),
                                    nudge_tensor(v, offsets, init_kv_cache_len, seq_order=2)))
        self.past_key_values = tuple(past_key_values)
