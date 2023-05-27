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
from typing import Tuple

import torch

from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.batch import Batch, ContrastiveBatch
from djl_python.scheduler.step_generation import greedy_step_generate, contrastive_step_generate
from djl_python.scheduler.utils import compute_offsets, compute_position_ids, compute_attention_mask, \
    assemble_prefix_kv_cache


class ContrastiveSeqBatchScheduler(SeqBatchScheduler):
    def init_forward(self, input_ids: torch.Tensor,
                     request_ids: torch.Tensor,
                     kv_cache: Tuple = None,
                     save_kv_cache_path="") -> SeqBatcher:
        initial_offsets = compute_offsets(input_ids, self.config)
        attention_mask = compute_attention_mask(input_ids, self.config)
        position_ids = compute_position_ids(input_ids.shape[0], input_ids.shape[1], initial_offsets, past_seq_len=0,
                                            repeat_offset=1)

        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(input_ids, position_ids,
                                                                                           attention_mask, kv_cache)

        model_input = [input_ids, position_ids, attention_mask]
        logits, past_key_values, past_hidden_states = self.lm_block.forward(model_input, past_key_values=kv_cache)

        # Create SeqBatcher
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        if kv_cache is not None:
            input_ids = torch.cat([dummy_input_ids, input_ids], dim=1)

            past_hidden_states = torch.concat([torch.zeros(input_ids.shape[0],
                                                           kv_cache[0][0].shape[2],
                                                           past_hidden_states.shape[-1],
                                                           dtype=past_hidden_states.dtype),
                                               past_hidden_states], dim=1)

        batch = ContrastiveBatch(
            past_output_ids=input_ids,
            past_attention_mask=attention_mask,
            past_hidden_states=past_hidden_states,
            past_key_values=past_key_values,
            logits=logits[:, -1, :]
        )

        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(initial_offsets, kv_cache[0][0].shape[2])

        return SeqBatcher(batch, request_ids, initial_offsets)

    def inference_call(self) -> torch.Tensor:
        batch = self.seq_batcher.batch
        # [batch, topK]
        top_k_ids = torch.topk(batch.logits, k=self.config.topk, dim=-1, largest=True, sorted=False).indices

        '''
        Prepare candidate model input
        '''
        # [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
        candidate_input_ids = torch.flatten(top_k_ids).view(-1, 1)
        assert candidate_input_ids.dtype == torch.int64
        assert len(candidate_input_ids.shape) == 2

        # [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
        k_copy_past_key_values = []
        for k, v in batch.past_key_values:
            k_new = torch.repeat_interleave(k, dim=0, repeats=self.config.topk)
            v_new = torch.repeat_interleave(v, dim=0, repeats=self.config.topk)
            k_copy_past_key_values.append((k_new, v_new))
        k_copy_past_key_values = tuple(k_copy_past_key_values)

        # [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
        batch_size = top_k_ids.shape[0]
        k_copy_past_attention_mask = torch.repeat_interleave(batch.past_attention_mask, dim=0, repeats=self.config.topk)
        k_copy_past_attention_mask = torch.concat([k_copy_past_attention_mask, torch.ones(batch_size *
                                                                                          self.config.topk, 1,
                                                                                          dtype=torch.int64)], dim=1)
        candidate_position_ids = compute_position_ids(candidate_input_ids.shape[0], candidate_input_ids.shape[1],
                                                      self.seq_batcher.offsets,
                                                      past_seq_len=self.seq_batcher.seq_len,
                                                      repeat_offset=self.config.topk)

        candidate_logits, candidate_past_key_values, candidate_hidden_states = self.lm_block.forward([
            candidate_input_ids,
            candidate_position_ids,
            k_copy_past_attention_mask],
            k_copy_past_key_values)

        output_ids, select = contrastive_step_generate(top_k_ids=top_k_ids,
                                                       logits=batch.logits,
                                                       context_hidden_states=batch.past_hidden_states,
                                                       top_k_hidden_states=candidate_hidden_states,
                                                       offsets=self.seq_batcher.offsets,
                                                       alpha=self.config.alpha)

        '''
        Select from the topk candidates and generate output and the new batch
        '''
        logits_dim = batch.logits.shape[1]
        _, num_heads, _, kv_dim = batch.past_key_values[0][0].shape
        past_seq_len = self.seq_batcher.seq_len
        hidden_dim = batch.past_hidden_states.shape[-1]

        # [batch, 1]
        a_range = torch.arange(batch_size)
        next_logits = candidate_logits.view(batch_size, self.config.topk, logits_dim)[a_range, select]

        next_past_key_values = []
        for k, v in candidate_past_key_values:
            k_new = k.view(batch_size, self.config.topk, num_heads, past_seq_len + 1, kv_dim)[a_range, select]
            v_new = v.view(batch_size, self.config.topk, num_heads, past_seq_len + 1, kv_dim)[a_range, select]
            next_past_key_values.append((k_new, v_new))
        next_past_key_values = tuple(next_past_key_values)

        delta_hidden_states = candidate_hidden_states.view(batch_size, self.config.topk, 1, hidden_dim)[a_range, select]
        next_hidden_states = torch.concat([batch.past_hidden_states, delta_hidden_states], dim=1)

        next_past_output_ids = torch.concat([batch.past_output_ids, output_ids], dim=1)
        next_past_attention_mask = torch.concat([batch.past_attention_mask, torch.ones((batch_size, 1),
                                                                                       dtype=torch.int64)], dim=1)
        self.seq_batcher.seq_len += 1
        self.seq_batcher.batch = ContrastiveBatch(past_output_ids=next_past_output_ids,
                                                  past_attention_mask=next_past_attention_mask,
                                                  past_hidden_states=next_hidden_states,
                                                  past_key_values=next_past_key_values,
                                                  logits=next_logits)

        # Exit
        self.seq_batcher.exit_criteria(output_ids, self.config.max_gen_seqlen, self.config.eos_token_id)

        return output_ids


class GreedySeqBatchScheduler(SeqBatchScheduler):

    def init_forward(self,
                     input_ids,
                     request_ids,
                     kv_cache: Tuple = None,
                     save_kv_cache_path=None) -> SeqBatcher:
        if input_ids.shape[0] != request_ids.shape[0] or len(
                request_ids.shape) != 2:
            raise Exception(
                "request_ids.shape does not match input_ids.shape or is illegal"
            )

        init_offsets = compute_offsets(input_ids, self.config)
        attention_mask = compute_attention_mask(input_ids, self.config)
        position_ids = compute_position_ids(input_ids.shape[0], input_ids.shape[-1], init_offsets, past_seq_len=0,
                                            repeat_offset=1)

        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(
            input_ids, position_ids, attention_mask, kv_cache)

        # output: list(logits, past_kv, hidden_state), where logits: [batch, sequence, vocab_dim]
        model_input = [input_ids, position_ids, attention_mask]
        logits, past_key_values, _ = self.lm_block.forward(model_input, past_key_values=kv_cache)

        # Create SeqBatcher
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        output_ids = input_ids if not kv_cache else torch.cat(
            [dummy_input_ids, input_ids], dim=1)

        batch = Batch(past_output_ids=output_ids, past_attention_mask=attention_mask, past_key_values=past_key_values,
                      logits=logits[:, -1, :])

        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(init_offsets, kv_cache[0][0].shape[2])

        return SeqBatcher(batch, request_ids, init_offsets)

    def inference_call(self) -> torch.Tensor:
        batch = self.seq_batcher.batch

        # [batch, seq=1]
        output_ids = greedy_step_generate(batch.logits)
        assert len(output_ids.shape) == 2

        # prepare the next model_input
        position_ids = compute_position_ids(output_ids.shape[0], output_ids.shape[-1], self.seq_batcher.offsets,
                                            past_seq_len=self.seq_batcher.seq_len,
                                            repeat_offset=1)
        past_attention_mask = torch.cat([
            batch.past_attention_mask,
            torch.ones_like(output_ids, dtype=torch.int64)
        ],
            dim=1)

        # Forward pass
        logits, past_key_values, _ = self.lm_block.forward([output_ids, position_ids, past_attention_mask],
                                                           past_key_values=batch.past_key_values)

        # Create SeqBatcher
        last_logits = logits[:, -1, :]  # logits: [batch, sequence, vocab_dim]
        past_output_ids = torch.cat([batch.past_output_ids, output_ids], dim=1)
        self.seq_batcher.batch = Batch(past_output_ids=past_output_ids, past_attention_mask=past_attention_mask,
                                       past_key_values=past_key_values, logits=last_logits)
        self.seq_batcher.seq_len += 1

        # Exit check
        self.seq_batcher.exit_criteria(output_ids, self.config.max_gen_seqlen,
                                       self.config.pad_token_id)

        return output_ids
