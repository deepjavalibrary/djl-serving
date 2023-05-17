from typing import List

import torch
from __future__ import annotations

from djl_python.scheduler.batch import Batch

PAD_TOKEN_ID = 220


class ContrastiveBatch(Batch):

    def __init__(
        self,
        seq_dim_order: List[int],
        past_output_ids: torch.Tensor = None,
        past_attention_mask: torch.Tensor = None,
        past_hidden_states: torch.Tensor = None,
        logits: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
    ):
        self.past_hidden_states = past_hidden_states
        self.logits = logits

        super().__init__(seq_dim_order,
                         past_output_ids=past_output_ids,
                         past_attention_mask=past_attention_mask,
                         past_key_values=past_key_values)

    def merge(self, batch: ContrastiveBatch, seq_delta) -> ContrastiveBatch:
        pass

    def trim(self, trim_sequence: int, keep_indices: List[int]):
        pass
