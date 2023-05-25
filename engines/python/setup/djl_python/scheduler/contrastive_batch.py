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
from typing import List

import torch

from djl_python.scheduler.batch import Batch


class ContrastiveBatch(Batch):

    def __init__(
            self,
            past_output_ids: torch.Tensor = None,
            past_attention_mask: torch.Tensor = None,
            past_hidden_states: torch.Tensor = None,
            logits: torch.Tensor = None,
            past_key_values: List[torch.Tensor] = None,
    ):
        self.past_hidden_states = past_hidden_states
        self.logits = logits

        super().__init__(past_output_ids=past_output_ids,
                         past_attention_mask=past_attention_mask,
                         past_key_values=past_key_values)

    def merge(self, batch: ContrastiveBatch, seq_delta) -> ContrastiveBatch:
        pass

    def trim(self, trim_sequence: int, keep_indices: List[int]):
        pass

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor, init_kv_cache_len: int):
        pass
