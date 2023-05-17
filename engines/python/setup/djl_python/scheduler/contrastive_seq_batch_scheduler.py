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
import torch

from djl_python.scheduler.contrastive_batch import ContrastiveBatch
from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher


class ContrastiveSeqBatchScheduler(SeqBatchScheduler):


    def init_forward(self, input_ids: torch.Tensor, batch_uids: torch.Tensor, config: SearchConfig) -> SeqBatcher:
        initial_offsets = SeqBatchScheduler.compute_offsets(input_ids, config)
        attention_mask = SeqBatchScheduler.compute_attention_mask(input_ids, config)
        position_ids = SeqBatchScheduler.compute_position_ids(input_ids, initial_offsets, 0, 1)

        # TODO: later add forward method here after LMBlock is implemented
        output = None

        last_logits = output.logits[:, -1, :]

        seq_dim_order = [1 for _ in range(3)]
        seq_dim_order.append(-1)
        seq_dim_order.extend([2 for _ in range(4, 28)])

        batch = ContrastiveBatch(
            seq_dim_order=seq_dim_order,
            past_output_ids=input_ids,
            past_attention_mask=attention_mask,
            past_key_values=output.past_key_values,
            logits=last_logits,
        )

        seq_batcher = SeqBatcher(batch, batch_uids, initial_offsets)
        return seq_batcher

    def inference_call(self) -> torch.Tensor:
        logits = self.seq_batcher.get_batch().logits
        top_k_ids = torch.topk(logits, k=self.config.k, dim=-1, largest=True, sorted=False)[1]
        batch = self.seq_batcher.get_batch()

        candidate_input_ids = torch.flatten(top_k_ids).reshape(-1, 1)
        assert candidate_input_ids.dtype == torch.int64
        assert len(candidate_input_ids.shape) == 2
