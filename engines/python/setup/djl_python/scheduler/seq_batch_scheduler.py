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
from typing import List, Union, Tuple, Dict

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
        self.search_configs = None

    @abstractmethod
    def init_forward(self, input_ids: torch.Tensor, batch_uids: torch.Tensor,
                     kv_cache: Union[Tuple, None]) -> SeqBatcher:
        pass

    def increment_forward(self, count: int) -> torch.Tensor:
        i = 0
        while i < count:
            if self.seq_batcher is None or self.seq_batcher.batch is None:
                print(
                    f"SeqBatcher not set. Please call add_batch. Current inference order is {i}"
                )
                break

            yield self.inference_call()

            if self.seq_batcher.seq_complete():
                self.results.update(self.seq_batcher.collect_and_trim())
            i += 1

    @abstractmethod
    def inference_call(self) -> torch.Tensor:
        pass

    def add_request(self, batch_uids: torch.Tensor,
                    input_ids: torch.Tensor,
                    search_configs: Dict[int, SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None):
        # Each request may have their own search_config
        # input1={‘text’:‘Hello’, ‘parameters’: {‘temperature’:1}}
        # input2={‘text’:’Goodbye, ‘parameters’: {‘temperature’:0.5}}
        new_seq_batcher = self.init_forward(input_ids, batch_uids, kv_cache)
        if self.seq_batcher:
            self.seq_batcher.add_batch(new_seq_batcher)
            self.search_configs.update(search_configs if search_configs else {})
        else:
            self.seq_batcher = new_seq_batcher
            self.search_configs = search_configs if search_configs else {}

    def collect_results(self):
        output = self.results
        self.results = {}
        for batch_uid in output:
            self.search_configs.pop(batch_uid, None)
        return output
