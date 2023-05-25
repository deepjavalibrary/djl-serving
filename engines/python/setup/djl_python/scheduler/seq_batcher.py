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

from typing import Dict, Union

from djl_python.scheduler.batch import Batch
import torch


class SeqBatcher(object):

    def __init__(self, batch: Batch, request_uids: torch.Tensor,
                 offsets: torch.Tensor):
        self.batch = batch
        self.request_uids = request_uids
        self.offsets = offsets
        self.exit_index_end_position = {}

        past_key_values_size = batch.past_attention_mask.size()
        self.batch_size = past_key_values_size[0]
        self.seq_len = past_key_values_size[1]

    def add_batch(self, seq_batcher: SeqBatcher):
        return self.merge_symmetric(self, seq_batcher)

    def merge_symmetric(self, seq_batcher1: SeqBatcher, seq_batcher2: SeqBatcher):
        seq_delta = seq_batcher1.seq_len - seq_batcher2.seq_len
        if seq_delta < 0:
            seq_batcher1, seq_batcher2 = seq_batcher2, seq_batcher1
            seq_delta = -seq_delta

        self.batch = seq_batcher1.batch.merge(seq_batcher2.batch, seq_delta)

        # update other batch control variables
        self.batch_size = seq_batcher1.batch_size + seq_batcher2.batch_size
        self.request_uids = torch.cat(
            [seq_batcher1.request_uids, seq_batcher2.request_uids], dim=0)
        self.offsets = torch.cat(
            [seq_batcher1.offsets, seq_batcher2.offsets + seq_delta], dim=0)
        self.seq_len = max(seq_batcher1.seq_len, seq_batcher2.seq_len)

    def collect_and_trim(self) -> Union[Dict[int, torch.Tensor], None]:
        if len(self.exit_index_end_position) == 0:
            return None

        finished_sequences = {}

        # collect the finished requests to the finished_sequences
        exit_indices = set()
        for batch_index, seq_end_position in self.exit_index_end_position.items(
        ):
            uid = self.request_uids[batch_index].item()
            offset = self.offsets[batch_index]
            output = self.batch.past_output_ids[batch_index,
                                                offset:seq_end_position]
            finished_sequences[uid] = output
            exit_indices.add(batch_index)

        # find the batch indices of the non-finished requests.
        keep_indices = torch.tensor(
            list(set(range(self.batch_size)) - exit_indices),
            dtype=torch.int64)

        # if all the requests finished generating sequences, then reset the batch and return
        if len(keep_indices) == 0:
            self.request_uids = torch.empty([0, 1],
                                            dtype=self.request_uids.dtype)
            self.offsets = torch.empty([0, 1], dtype=self.offsets.dtype)
            self.batch = None
            self.batch_size = 0
            self.seq_len = 0
        else:
            self.request_uids = self.request_uids[keep_indices]
            self.offsets = self.offsets[keep_indices]
            trim_seq_len = torch.min(self.offsets, dim=0).values.item()
            self.offsets = self.offsets - trim_seq_len

            self.batch.trim(keep_indices, trim_seq_len)
            self.batch_size -= len(exit_indices)
            self.seq_len -= trim_seq_len

        self.exit_index_end_position = {}

        return finished_sequences

    def exit_criteria(self, output_ids: torch.Tensor, max_length: int,
                      eos_token_id: int):
        for i in range(len(output_ids)):
            if self.seq_len - self.offsets[i] >= max_length or output_ids[
                    i] == eos_token_id:
                if i not in self.exit_index_end_position:
                    self.exit_index_end_position[i] = self.seq_len

    def seq_complete(self) -> bool:
        return len(self.exit_index_end_position) > 0
