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
from abc import ABC, abstractmethod

from djl_python.scheduler.batch import Batch, ContrastiveBatch
from djl_python.scheduler.lm_block import LMBlock
import torch
from djl_python.scheduler import SearchConfig


class SeqBatcher(ABC):
    """
    This is a data class, which stores the search state (Batch), the control variables (eg seq_len, offsets, etc),
    and batch operations like merge_batch, trim, init_forward and inference_call. The latter two are search algorithm
    specific. Users may provide their own autoregressive searching algorithm by inheriting this class and overwriting
    the init_forward, inference_call along with the corresponding Batch.
    """

    def __init__(self, batch: Batch, request_uids: torch.Tensor,
                 offsets: torch.Tensor,
                 search_configs: defaultdict[Any,
                                             SearchConfig], lm_block: LMBlock):
        # Utility variables
        self.lm_block = lm_block
        self.exit_index = set()

        # Variables updated in a batch operation
        self.batch = batch
        self.request_uids = request_uids
        self.offsets = offsets
        self.search_configs = search_configs
        self.search_config_list_cache = None
        self.batch_size, _, self.seq_len, _ = batch.past_key_values[0][0].size(
        )

        # Used in GreedySeqBatcher
        # This is cached output of sampler_bucket_sort result used through inferences.
        self.sampler_bucket_sort_cache: Union[Tuple[Dict[str, torch.tensor],
                                                    List[SearchConfig],
                                                    List[SearchConfig]], None] = None

    @classmethod
    @abstractmethod
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            kv_cache_input_ids: Union[torch.tensor, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[int]]]:
        pass

    @staticmethod
    @abstractmethod
    def _get_batch_cls():
        pass

    @abstractmethod
    def forward(self) -> List[List[int]]:
        pass

    @torch.no_grad()
    def add_batch(self, seq_batcher: SeqBatcher):
        if self.lm_block != seq_batcher.lm_block:
            raise "lm_blocks are not the same instance, not mergable"

        self._merge_symmetric(self, seq_batcher)

    def _merge_symmetric(self, seq_batcher1: SeqBatcher,
                         seq_batcher2: SeqBatcher):
        seq_delta = seq_batcher1.seq_len - seq_batcher2.seq_len
        if seq_delta < 0:
            seq_batcher1, seq_batcher2 = seq_batcher2, seq_batcher1
            seq_delta = -seq_delta

        # merge batches
        self.batch = seq_batcher1.batch.merge(seq_batcher2.batch, seq_delta)

        # update other batch control variables
        self.batch_size = seq_batcher1.batch_size + seq_batcher2.batch_size
        self.request_uids = torch.cat(
            [seq_batcher1.request_uids, seq_batcher2.request_uids], dim=0)
        self.offsets = torch.cat(
            [seq_batcher1.offsets, seq_batcher2.offsets + seq_delta], dim=0)
        self.seq_len = max(seq_batcher1.seq_len, seq_batcher2.seq_len)
        seq_batcher1.search_configs.update(seq_batcher2.search_configs)
        self.search_configs = seq_batcher1.search_configs

        self.search_config_list_cache = None
        self.sampler_bucket_sort_cache = None

    @torch.no_grad()
    def collect_and_trim(self) -> List[int]:
        if len(self.exit_index) == 0:
            return []

        exit_request_uids = []
        keep_indices_list = []
        request_uids_list = self.request_uids.view(-1).tolist()
        for i in range(self.batch_size):
            if i in self.exit_index:
                exit_request_uids.append(request_uids_list[i])
            else:
                keep_indices_list.append(i)
        keep_indices = torch.tensor(keep_indices_list,
                                    dtype=torch.int64,
                                    device=self.offsets.device)

        # if all the requests finished generating sequences, then reset the batch and return
        if len(keep_indices) == 0:
            self.request_uids = torch.empty([0, 1],
                                            dtype=self.request_uids.dtype,
                                            device=self.request_uids.device)
            self.offsets = torch.empty([0, 1],
                                       dtype=self.offsets.dtype,
                                       device=self.offsets.device)
            self.search_configs.clear()

            self.batch = None
            self.batch_size = 0
            self.seq_len = 0
        else:
            for idx in self.exit_index:
                del self.search_configs[self.request_uids[idx].item()]
            self.request_uids = self.request_uids[keep_indices]
            self.offsets = self.offsets[keep_indices]
            trim_seq_len = torch.min(self.offsets, dim=0).values.item()
            self.offsets.sub_(trim_seq_len)

            self.batch = self.batch.trim(keep_indices, trim_seq_len)
            self.batch_size -= len(self.exit_index)
            self.seq_len -= trim_seq_len

        self.search_config_list_cache = None
        self.sampler_bucket_sort_cache = None
        self.exit_index = set()
        return exit_request_uids

    def exit_criteria(self, output_ids: torch.Tensor, search_configs):
        for i, (output_id, request_uid, offset) in enumerate(
                zip(
                    output_ids.view(-1).tolist(),
                    self.request_uids.view(-1).tolist(),
                    self.offsets.view(-1).tolist())):
            if self.seq_len - offset >= search_configs[request_uid].max_seqlen \
                    or output_id == search_configs[request_uid].eos_token_id:
                if i not in self.exit_index:
                    self.exit_index.add(i)

    def seq_complete(self) -> bool:
        return len(self.exit_index) > 0

    def is_empty(self) -> bool:
        return self.batch is None

    @torch.no_grad()
    def split(self, partitions: List[List[int]]) -> List[SeqBatcher]:
        result = []
        for partition in partitions:
            if len(partition) == 0:
                continue
            keep_indices = torch.tensor(partition,
                                        dtype=torch.int64,
                                        device=self.offsets.device)
            request_uids = self.request_uids[keep_indices]
            offsets = self.offsets[keep_indices]
            trim_seq_len = torch.min(self.offsets, dim=0).values.item()
            batch = self.batch.trim(keep_indices, trim_seq_len)

            search_configs = defaultdict(self.search_configs.default_factory)
            search_configs.update({
                key: self.search_configs[key]
                for key in request_uids.view(-1).tolist()
            })

            result.append(
                self.__class__(batch, request_uids, offsets, search_configs,
                               self.lm_block))

        return result
