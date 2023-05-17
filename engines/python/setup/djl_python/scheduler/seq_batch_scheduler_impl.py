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
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.batch import Batch
from djl_python.scheduler.step_generation import greedy_step_generate

class ContrastiveSeqBatchScheduler(SeqBatchScheduler):
    def init_forward(self, input_ids: torch.Tensor, request_ids: torch.Tensor, kv_cache=None, save_kv_cache_path="") -> \
            SeqBatcher:

        initial_offsets = SeqBatchScheduler.compute_offsets(input_ids, self.config)
        attention_mask = SeqBatchScheduler.compute_attention_mask(input_ids, self.config)
        position_ids = SeqBatchScheduler.compute_position_ids(input_ids, initial_offsets, past_seq_len=0, repeat=1)

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

        seq_batcher = SeqBatcher(batch, request_ids, initial_offsets)
        return seq_batcher

    def inference_call(self) -> torch.Tensor:
        logits = self.seq_batcher.batch.logits
        top_k_ids = torch.topk(logits, k=self.config.k, dim=-1, largest=True, sorted=False)[1]
        batch = self.seq_batcher.batch

        candidate_input_ids = torch.flatten(top_k_ids).reshape(-1, 1)
        assert candidate_input_ids.dtype == torch.int64
        assert len(candidate_input_ids.shape) == 2


class GreedySeqBatchScheduler(SeqBatchScheduler):

    def init_forward(self, input_ids, request_ids, kv_cache=None, save_kv_cache_path="") -> SeqBatcher:
        if input_ids.shape[0] != request_ids.shape[0] or len(request_ids.shape) != 2:
            raise Exception("request_ids.shape does not match input_ids.shape or is illegal")

        init_offsets = self.compute_offsets(input_ids, self.config)
        attention_mask = self.compute_attention_mask(input_ids, self.config)
        position_ids = self.compute_position_ids(input_ids, init_offsets, past_seq_len=0, repeat=1)

        dummy_input_ids = None
        if kv_cache is not None:
            if input_ids.shape[0] > 1  or kv_cache[0][0].shape[0] > 1:
                raise Exception("When kv_cache that precedes the input_ids is provided, the init_forward is restricted"
                                " to process one sequence at a time, which is not padded. This avoids the padding "
                                "bubble between the precedent kv_cache and the input_ids.")
            if torch.any(init_offsets != 0):
                raise Exception("When kv_cache which precedes the input_ids is provided, the input_ids shouldn't be "
                                "padded")
            kv_cache_seqlen = kv_cache[0][0].shape[2]
            attention_mask = torch.cat([torch.ones((1, kv_cache_seqlen), dtype=torch.int64), attention_mask], dim=1)
            position_ids += kv_cache_seqlen
            dummy_input_ids = torch.full([1, kv_cache_seqlen], fill_value=0, dtype=input_ids.dtype)

        # output: list(logits, past_kv, hidden_state), where logits: [batch, sequence, vocab_dim]
        model_input = [input_ids, position_ids, attention_mask]
        output = self.lm_block.forward(model_input, past_key_values=kv_cache)

        # Create SeqBatcher
        last_logits = output[0][:, -1, :]
        past_key_values = output[1]
        if len(save_kv_cache_path) > 0:
            torch.save(output[1], save_kv_cache_path)
        batch = Batch(seq_dim_order=[],
                      past_output_ids=input_ids if not kv_cache else torch.cat([dummy_input_ids, input_ids], dim=1),
                      past_attention_mask=attention_mask,
                      logits=last_logits,
                      past_key_values=past_key_values)
        return SeqBatcher(batch, request_ids, init_offsets)

    def inference_call(self) -> torch.Tensor:
        batch = self.seq_batcher.batch

        # [batch, seq=1]
        output_ids = greedy_step_generate(batch.logits)
        assert len(output_ids.shape) == 2

        # prepare the next model_input
        model_input = []
        model_input.append(output_ids)
        model_input.append(self.compute_position_ids(output_ids, self.seq_batcher.offsets, past_seq_len=self.seq_batcher.seq_len,
                                                     repeat=1))
        past_attention_mask = batch.past_attention_mask
        past_attention_mask = torch.cat([past_attention_mask, torch.ones_like(output_ids, dtype=torch.int64)], dim=1)
        model_input.append(past_attention_mask)

        # output: list(logits, past_kv, hidden_state), where logits: [batch, sequence, vocab_dim]
        output = self.lm_block.forward(model_input, past_key_values=batch.past_key_values)

        # Create SeqBatcher
        last_logits = output[0][:, -1, :]
        past_key_values = output[1]
        past_output_ids = torch.cat([batch.past_output_ids, output_ids], dim=1)
        self.seq_batcher.batch = Batch(seq_dim_order=[],
                      past_output_ids=past_output_ids,
                      past_attention_mask=past_attention_mask,
                      logits=last_logits,
                      past_key_values=past_key_values)
        self.seq_batcher.seq_len += 1

        # exit check
        self.seq_batcher.exit_criteria(output_ids, self.config.max_seq_length, self.config.pad_token_id)

        return output_ids


