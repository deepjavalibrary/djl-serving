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

    def get_batch(self) -> Batch:
        return self.batch

    def add_batch(self, new_seq_batcher: SeqBatcher):
        pass

    def exit_criteria(self, output_ids: torch.Tensor, max_length: int,
                      eos_token_id: int):
        pass

    def collect_and_trim(self) -> Dict[int, torch.Tensor]:
        pass

    def seq_complete(self) -> bool:
        pass
