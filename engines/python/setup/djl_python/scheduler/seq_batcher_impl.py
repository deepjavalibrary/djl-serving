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

from collections import defaultdict
from typing import Dict, Union, Tuple, List, Any

from djl_python.scheduler.batch import Batch, ContrastiveBatch
from djl_python.scheduler.lm_block import LMBlock
import torch
from torch.nn.functional import normalize, softmax

from djl_python.scheduler.step_generation import greedy_step_generate, contrastive_step_generate
from djl_python.scheduler.utils import compute_offsets, compute_attention_mask, compute_position_ids, \
    assemble_prefix_kv_cache
from djl_python.scheduler import SearchConfig
import warnings

from djl_python.scheduler.seq_batcher import SeqBatcher


class GreedySeqBatcher(SeqBatcher):

    @classmethod
    @torch.no_grad()
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            kv_cache_input_ids: Union[torch.tensor, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[int]]]:

        if input_ids.shape[0] != request_uids.shape[0] or len(
                request_uids.shape) != 2:
            raise Exception(
                "request_uids.shape does not match input_ids.shape or is illegal"
            )

        initial_offsets = compute_offsets(input_ids, [
            search_configs[r].pad_token_id
            for r in request_uids.view(-1).tolist()
        ])
        attention_mask = compute_attention_mask(initial_offsets,
                                                input_ids.shape[-1])
        position_ids = compute_position_ids(input_ids.shape[0],
                                            input_ids.shape[1],
                                            initial_offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        # Handle the kv_cache
        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(
            input_ids, position_ids, attention_mask, kv_cache,
            kv_cache_input_ids)

        # Forward call
        model_input = [input_ids, position_ids, attention_mask]
        lm_output = lm_block.forward(*model_input, past_key_values=kv_cache)
        logits, past_key_values = lm_output.logits, lm_output.past_key_values
        last_logits = logits[:, -1, :]

        # Save kv_cache of input_ids
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        # Generate next token and batch
        next_input_ids = greedy_step_generate(
            last_logits).indices  # [batch, 1]
        batch = Batch(next_input_ids=next_input_ids,
                      past_key_values=past_key_values)
        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(initial_offsets,
                                                  kv_cache[0][0].shape[2])

        # Output
        output_ids_list = []
        for i, (input_id,
                offset) in enumerate(zip(input_ids.tolist(), initial_offsets)):
            to_append = input_id[offset:]
            if kv_cache is not None:
                to_append = dummy_input_ids[i].tolist() + to_append
            output_ids_list.append(to_append)

        return cls(batch, request_uids, initial_offsets, search_configs,
                   lm_block), output_ids_list

    @torch.no_grad()
    def forward(self) -> List[List[int]]:
        batch = self.batch

        # [batch, seq=1]
        output_ids = batch.next_input_ids
        assert len(output_ids.shape) == 2

        # Prepare the next model_input
        position_ids = compute_position_ids(output_ids.shape[0],
                                            output_ids.shape[-1],
                                            self.offsets,
                                            past_seq_len=self.seq_len,
                                            repeat_offset=1)

        past_attention_mask = compute_attention_mask(self.offsets,
                                                     self.seq_len + 1)

        # Forward pass
        lm_output = self.lm_block.forward(
            output_ids,
            position_ids,
            past_attention_mask,
            past_key_values=batch.past_key_values)
        logits, past_key_values = lm_output.logits, lm_output.past_key_values

        # Create SeqBatcher
        last_logits = logits[:, -1, :]  # logits: [batch, sequence, vocab_dim]
        next_input_ids = greedy_step_generate(
            last_logits).indices  # [batch, 1]
        self.batch = self._get_batch_cls()(past_key_values=past_key_values,
                                           next_input_ids=next_input_ids)
        self.seq_len += 1

        # Exit check
        self.exit_criteria(output_ids, self.search_configs)

        return output_ids.tolist()

    @staticmethod
    def _get_batch_cls():
        return Batch


class ContrastiveSeqBatcher(SeqBatcher):

    @classmethod
    @torch.no_grad()
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            kv_cache_input_ids: Union[torch.tensor, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[int]]]:

        if input_ids.shape[0] != request_uids.shape[0] or len(
                request_uids.shape) != 2:
            raise Exception(
                "request_uids.shape does not match input_ids.shape or is illegal"
            )

        initial_offsets = compute_offsets(input_ids, [
            search_configs[r].pad_token_id
            for r in request_uids.view(-1).tolist()
        ])
        attention_mask = compute_attention_mask(initial_offsets,
                                                input_ids.shape[-1])
        position_ids = compute_position_ids(input_ids.shape[0],
                                            input_ids.shape[1],
                                            initial_offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        # Handle the kv_cache
        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(
            input_ids, position_ids, attention_mask, kv_cache,
            kv_cache_input_ids)

        # Forward call
        model_input = [input_ids, position_ids, attention_mask]
        lm_output = lm_block.forward(*model_input, past_key_values=kv_cache)
        logits, past_key_values, past_hidden_states = lm_output[
            'logits'], lm_output['past_key_values'], lm_output[
                'hidden_states'][0]

        last_logits = logits[:, -1, :]

        # Save kv_cache of input_ids
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        # ---- Specific to contrastive search ----#
        if kv_cache is not None and kv_cache_input_ids is None:
            warnings.warn(
                "You input a kv_cache but didn't provide the corresponding kv_cache_input_ids. In "
                "contrastive search, the result will depend on this input_ids. In the following, "
                "the result is obtained by assuming kv_cache_input_ids are all 0."
            )

        if kv_cache is not None:
            past_hidden_states = torch.concat([
                torch.zeros(input_ids.shape[0],
                            kv_cache[0][0].shape[2],
                            past_hidden_states.shape[-1],
                            dtype=past_hidden_states.dtype,
                            device=past_hidden_states.device),
                past_hidden_states
            ],
                                              dim=1)

        # Generate next token and batch
        topk = search_configs["non_exist_key"].topk
        # [batch, vocab_size=50257]
        last_probs = softmax(last_logits, dim=1)
        # [batch, topk]
        top_k_probs, top_k_ids = greedy_step_generate(last_probs, topk)
        batch = cls._get_batch_cls()(next_input_ids=top_k_ids,
                                     past_key_values=past_key_values,
                                     past_hidden_states=past_hidden_states,
                                     top_k_probs=top_k_probs)
        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(initial_offsets,
                                                  kv_cache[0][0].shape[2])
        # Output ids
        output_ids_list = []
        for i, (input_id,
                offset) in enumerate(zip(input_ids.tolist(), initial_offsets)):
            to_append = input_id[offset:]
            if kv_cache is not None:
                to_append = dummy_input_ids[i].tolist() + to_append
            output_ids_list.append(to_append)

        return cls(batch, request_uids, initial_offsets, search_configs,
                   lm_block), output_ids_list

    @torch.no_grad()
    def forward(self) -> List[List[int]]:
        batch = self.batch
        config = self.search_configs["non_exist_key"]

        # [batch, topK]
        top_k_ids = batch.next_input_ids

        # Prepare candidate model input
        # [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
        candidate_input_ids = top_k_ids.view(-1, 1)
        assert candidate_input_ids.dtype == torch.int64
        assert len(candidate_input_ids.shape) == 2

        # [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
        k_copy_past_key_values = []
        for k, v in batch.past_key_values:
            k_new = torch.repeat_interleave(k, dim=0, repeats=config.topk)
            v_new = torch.repeat_interleave(v, dim=0, repeats=config.topk)
            k_copy_past_key_values.append((k_new, v_new))
        k_copy_past_key_values = tuple(k_copy_past_key_values)

        # [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
        batch_size = top_k_ids.shape[0]
        k_copy_past_attention_mask = compute_attention_mask(
            offsets=self.offsets,
            seq_len=self.seq_len + 1,
            repeat_offset=config.topk)
        candidate_position_ids = compute_position_ids(
            candidate_input_ids.shape[0],
            candidate_input_ids.shape[1],
            self.offsets,
            past_seq_len=self.seq_len,
            repeat_offset=config.topk)

        lm_output = self.lm_block.forward(candidate_input_ids,
                                          candidate_position_ids,
                                          k_copy_past_attention_mask,
                                          k_copy_past_key_values)
        # [batch * topK, ..., seq_past + 1, ...]
        candidate_logits, candidate_past_key_values, candidate_hidden_states = lm_output.logits, \
                                                                              lm_output.past_key_values, \
                                                                               lm_output.hidden_states[0]

        output_ids, select = contrastive_step_generate(
            top_k_ids=top_k_ids,
            top_k_probs=batch.top_k_probs,
            context_hidden_states=batch.past_hidden_states,
            top_k_hidden_states=candidate_hidden_states,
            offsets=self.offsets,
            alpha=config.alpha)

        # Select from the topk candidates and generate output and the new batch
        logits_dim = candidate_logits.shape[-1]
        _, num_heads, _, kv_dim = batch.past_key_values[0][0].shape
        past_seq_len = self.seq_len
        hidden_dim = batch.past_hidden_states.shape[-1]

        # [batch, 1]
        a_range = torch.arange(batch_size)
        next_logits = candidate_logits.view(batch_size, config.topk,
                                            logits_dim)[a_range, select]

        next_past_key_values = []
        for k, v in candidate_past_key_values:
            k_new = k.view(batch_size, config.topk, num_heads,
                           past_seq_len + 1, kv_dim)[a_range, select]
            v_new = v.view(batch_size, config.topk, num_heads,
                           past_seq_len + 1, kv_dim)[a_range, select]
            next_past_key_values.append((k_new, v_new))
        next_past_key_values = tuple(next_past_key_values)

        delta_hidden_states = candidate_hidden_states.view(
            batch_size, config.topk, 1, hidden_dim)[a_range, select]
        next_hidden_states = torch.concat(
            [batch.past_hidden_states, delta_hidden_states], dim=1)

        self.seq_len += 1

        # [batch, vocab_size]
        next_probs = softmax(next_logits, dim=1)
        # [batch, topk]
        top_k_probs, top_k_ids = greedy_step_generate(next_probs, config.topk)
        self.batch = ContrastiveBatch(next_input_ids=top_k_ids,
                                      past_key_values=next_past_key_values,
                                      past_hidden_states=next_hidden_states,
                                      top_k_probs=top_k_probs)

        # Exit
        self.exit_criteria(output_ids, self.search_configs)

        return output_ids.tolist()

    @staticmethod
    def _get_batch_cls():
        return ContrastiveBatch
