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
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, List

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.lm_block import LMBlock


class SeqBatchScheduler(ABC):

    def __init__(self, lm_block: LMBlock, default_config: SearchConfig):
        self.lm_block = lm_block
        self.results = {}
        self.seq_batcher = None
        self.config = default_config
        self.search_configs = defaultdict(lambda: default_config)

    @abstractmethod
    def init_forward(self, input_ids: torch.Tensor, request_uids: torch.Tensor,
                     kv_cache: Union[Tuple, None]) -> SeqBatcher:
        pass

    def is_empty(self):
        return self.seq_batcher is None or self.seq_batcher.batch is None

    def increment_forward(self, count: int) -> torch.Tensor:
        i = 0
        while i < count:
            if self.seq_batcher is None or self.seq_batcher.batch is None:
                print(
                    f"SeqBatcher not set. Please call add_batch. Current inference order is {i}"
                )
                break

            output_ids = self.inference_call()

            # collect output
            for request_uid, output_id in zip(self.seq_batcher.request_uids, output_ids):
                self.results[request_uid.item()].append(output_id.item())

            # trim the sequence batcher
            self.seq_batcher.collect_and_trim()
            i += 1

            yield output_ids

    @abstractmethod
    def inference_call(self) -> torch.Tensor:
        pass

    def add_request(self, request_uids: torch.Tensor,
                    input_ids: torch.Tensor,
                    search_configs: List[SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None):
        new_seq_batcher, output_ids = self.init_forward(input_ids, request_uids, kv_cache)
        for request_uid, output_id in zip(request_uids, output_ids):
            self.results[request_uid.item()] = output_id.tolist()

        if self.seq_batcher and self.seq_batcher.batch:
            self.seq_batcher.add_batch(new_seq_batcher)
        else:
            self.seq_batcher = new_seq_batcher

        if search_configs:
            for request, search_config in zip(request_uids, search_configs):
                self.search_configs[request.item()] = search_config

    def collect_results(self):
        output = self.results
        self.results = {}
        for request_uid in output:
            self.search_configs.pop(request_uid, None)
        return output
