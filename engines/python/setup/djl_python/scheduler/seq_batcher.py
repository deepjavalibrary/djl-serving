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

from typing import Dict

from djl_python.scheduler.batch import Batch
import torch


class SeqBatcher(object):

    def __init__(self, batch: Batch, request_uids: torch.Tensor,
                 offsets: torch.Tensor):
        self.batch = batch
        self.request_uids = request_uids
        self.offsets = offsets
        self.exit_index_end_position = {}

        output_ids_size = batch.past_output_ids.size()
        self.batch_size = output_ids_size.size[0]
        self.seq_len = output_ids_size[1]

    def get_batch(self) -> Batch:
        return self.batch

    def add_batch(self, new_seq_batcher: SeqBatcher):
        seq_delta = self.seq_len - new_seq_batcher.seq_len
        if seq_delta >= 0:
            self.batch = self.batch.merge(new_seq_batcher.batch, seq_delta)
        else:
            self.batch = new_seq_batcher.batch.merge(self.batch, -seq_delta)

        self.batch_size = self.batch_size + new_seq_batcher.batch_size
        self.request_uids = torch.cat(
            [self.request_uids, new_seq_batcher.request_uids], dim=0)
        self.offsets = torch.cat([self.offsets, new_seq_batcher.offsets],
                                 dim=0)

    def exit_criteria(self, output_ids: torch.Tensor, max_length: int,
                      eos_token_id: int):
        output_ids_list = output_ids.tolist()
        offsets_list = self.offsets.tolist()
        for i in range(len(output_ids_list)):
            if self.seq_len - offsets_list[i] >= max_length or output_ids_list[
                    i] == eos_token_id:
                if i not in self.exit_index_end_position:
                    self.exit_index_end_position[i] = self.seq_len

    def collect_and_trim(self) -> Dict[int, torch.Tensor]:
        if len(self.exit_index_end_position) == 0:
            return None

        finished_sequences = {}
        exit_indices = set()

        # Adds the finished requests to the finished_sequences
        # Batch index is added to the set to be removed from batch later.
        for batch_index, seq_end_position in self.exit_index_end_position.items(
        ):
            uid = self.request_uids[batch_index]
            offset = self.offsets[batch_index]
            output = self.batch.past_output_ids[batch_index,
                                                offset:seq_end_position]
            finished_sequences[uid] = output
            exit_indices.add(batch_index)

        # finding the row with non-finished sequences.
        keep_indices = []
        j = 0
        for i in range(self.batch_size):
            if i not in exit_indices:
                keep_indices[j] = i
                j += 1

        # if all the requests finished generating sequences, then reset the batch and return
        if len(keep_indices) == 0:
            self.request_uids = torch.zeros([0, 1],
                                            dtype=self.request_uids.dtype)
            self.offsets = torch.zeros([0, 1], dtype=self.offsets.dtype)
            self.batch = None
            self.batch_size = 0
            self.seq_len = 0
            self.exit_index_end_position = None
            return finished_sequences

        self.request_uids = torch.index_select(
            self.request_uids, 0,
            torch.LongTensor(keep_indices)).resize(-1, 1)
        self.offsets = torch.index_select(
            self.offsets, 0, torch.LongTensor(keep_indices)).resize(-1, 1)
        trim_sequence = torch.min(self.offsets, dim=0).tolist()[0]
        self.offsets = torch.subtract(self.offsets, trim_sequence)

        # Trim batch and sequence dimension if needed.

    def seq_complete(self) -> bool:
        return len(self.exit_index_end_position) > 0
