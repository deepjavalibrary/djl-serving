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
from abc import ABC, abstractmethod
from typing import List

import torch


class Batch(ABC):

    def __init__(self,
                 seq_dim_order: List[int],
                 past_output_ids: torch.Tensor = None,
                 past_attention_mask: torch.Tensor = None,
                 past_key_values: List[torch.Tensor] = None):
        self.seq_dim_order = seq_dim_order
        self.past_output_ids = past_output_ids
        self.past_key_values = past_key_values
        self.past_attention_mask = past_attention_mask

    def get_seq_dim_order(self) -> List[int]:
        return self.seq_dim_order

    # merges another batch with itself.
    @abstractmethod
    def merge(self, batch: Batch, seq_delta) -> Batch:
        pass

    @abstractmethod
    def trim(self, trim_sequence: int, keep_indices: List[int]):
        pass
