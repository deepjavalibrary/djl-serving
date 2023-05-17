import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher


class ContrastiveSeqBatchScheduler(SeqBatchScheduler):

    def init_forward(self, input_ids: torch.Tensor, batch_uids: torch.Tensor,
                     config: SearchConfig) -> SeqBatcher:
        pass

    def inference_call(self) -> torch.Tensor:
        pass
