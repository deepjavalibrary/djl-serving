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
from typing import List, Union

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.lm_block import LMBlock


class SeqBatchScheduler(ABC):

    def __init__(self, lm_block: LMBlock, config: SearchConfig):
        self.lm_block = lm_block
        self.results = {}
        self.seq_batcher = None
        self.config = config

    @abstractmethod
    def init_forward(self, input_ids: torch.Tensor,
                     batch_uids: torch.Tensor,
                     kv_cache: Union[List[torch.Tensor], None]) -> SeqBatcher:
        pass

    def increment_forward(self, count: int) -> torch.Tensor:
        i = 0
        intermediate = None
        while i < count:
            if self.seq_batcher is None or self.seq_batcher.batch is None:
                print(f"SeqBatcher not set. Please call add_batch. Current inference order is {i}")
                break

            yield self.inference_call()

            if self.seq_batcher.seq_complete():
                self.results.update(self.seq_batcher.collect_and_trim())
            i += 1

    @abstractmethod
    def inference_call(self) -> torch.Tensor:
        pass

    def add_request(self, input_ids, batch_uids, kv_cache=None):
        new_seq_batcher = self.init_forward(input_ids, batch_uids, kv_cache)
        if self.seq_batcher:
            self.seq_batcher.add_batch(new_seq_batcher)
        else:
            self.seq_batcher = new_seq_batcher

    def collect_results(self):
        output = self.results
        self.results = {}
        return output

    @staticmethod
    def compute_offsets(input_ids: torch.Tensor,
                        config: SearchConfig) -> torch.Tensor:
        num_batch = input_ids.shape[0]
        seq_size = input_ids.shape[1]

        offsets = []
        for i in range(num_batch):
            sequence = input_ids[i].tolist()
            index = 0
            while index < seq_size:
                if sequence[index] != config.pad_token_id:
                    break
                index += 1

            offsets.append(index)

        return torch.tensor(offsets, dtype=torch.int64).view(-1, 1)

    @staticmethod
    def compute_attention_mask(input_ids, config):
        num_batch = input_ids.shape[0]
        seq_size = input_ids.shape[1]

        # attention_mask
        attention_mask = torch.repeat_interleave(torch.ones(
            [1, input_ids.shape[-1]], dtype=torch.int64).reshape(1, -1),
                                                 dim=0,
                                                 repeats=num_batch)

        # Linear searches the offset and set the mask
        for i in range(num_batch):
            sequence = input_ids[i].tolist()
            index = 0
            while index < seq_size:
                if sequence[index] != config.pad_token_id:
                    break
                index += 1

            attention_mask[i][0:index] = 0

        return attention_mask

    @staticmethod
    def compute_position_ids(input_ids: torch.Tensor,
                             offsets: torch.Tensor,
                             past_seq_len: int,
                             repeat: int):
        position_range = torch.arange(
            start=past_seq_len,
            end=past_seq_len + input_ids.shape[-1],
            step=1,
            dtype=torch.int64).view(1, -1)

        position_ids = torch.repeat_interleave(position_range,
                                               dim=0,
                                               repeats=input_ids.shape[0])

        position_ids_shifted = position_ids - torch.repeat_interleave(
            offsets.view(-1, 1),
            dim=0,
            repeats=repeat)

        position_ids = torch.maximum(position_ids_shifted, torch.zeros_like(position_ids_shifted))
        return position_ids
