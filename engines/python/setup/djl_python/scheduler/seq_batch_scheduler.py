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
