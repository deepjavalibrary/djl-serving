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

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batcher import SeqBatcher


class SeqBatchScheduler(ABC):

    def __init__(self):
        self.lm_block = None
        self.results = {}
        self.seq_batcher = None
        self.config = SearchConfig()

    @abstractmethod
    def init_forward(self, input_ids: torch.Tensor, batch_uids: torch.Tensor,
                     config: SearchConfig) -> SeqBatcher:
        pass

    def increment_forward(self, count: int) -> torch.Tensor:
        pass

    @abstractmethod
    def inference_call(self) -> torch.Tensor:
        pass

    def add_request(self, input_ids, batch_uids, config):
        pass

    def collect_results(self):
        pass

    @staticmethod
    def compute_offsets(input_ids: torch.Tensor,
                        config: SearchConfig) -> torch.Tensor:
        pass

    @staticmethod
    def compute_attention_mask(input_ids, config):
        pass

    @staticmethod
    def compute_position_ids(input_ids: torch.Tensor, offsets: torch.Tensor,
                             past_seq_len: int, repeat: int):
        pass
