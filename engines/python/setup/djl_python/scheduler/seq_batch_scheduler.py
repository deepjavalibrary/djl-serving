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
from collections import defaultdict
from typing import Union, Tuple, List, Dict, Type

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.lm_block import LMBlock
from djl_python.scheduler.seq_batcher import SeqBatcher


class SeqBatchScheduler:
    """
    This is a scheduler that manages the SeqBatcher, providing API which allows for actions like addBatch,
    collectResults.
    """

    def __init__(self, lm_block: LMBlock,
                 default_seq_batcher_cls: Type[SeqBatcher],
                 default_config: SearchConfig):
        self.default_search_configs = defaultdict(lambda: default_config)
        self.default_seq_batcher_cls = default_seq_batcher_cls
        self.lm_block = lm_block
        self.results: Dict[int, List[int]] = defaultdict(list)

        self.seq_batchers: Dict[
                           Type[SeqBatcher]:List[SeqBatcher]] = defaultdict(
            list)  # {key: List[SeqBatcher]}

    def add_request(self,
                    input_ids: torch.Tensor,
                    request_uids: torch.Tensor,
                    seq_batcher_cls: Type[SeqBatcher] = None,
                    search_configs: List[SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None,
                    save_kv_cache_path: str = None):
        device = input_ids.device
        request_uids = request_uids.to(device)
        if kv_cache:
            kv_list = []
            for k, v in kv_cache:
                k_new = k.to(device)
                v_new = v.to(device)
                kv_list.append((k_new, v_new))
            kv_cache = tuple(kv_list)

        if search_configs:
            for request, search_config in zip(
                    request_uids.view(-1).tolist(), search_configs):
                self.default_search_configs[request] = search_config

        seq_batcher_cls = self.default_seq_batcher_cls if seq_batcher_cls is None else seq_batcher_cls

        # Prefill
        new_seq_batcher, output_ids = seq_batcher_cls.init_forward(
            input_ids=input_ids,
            request_uids=request_uids,
            lm_block=self.lm_block,
            search_configs=self.default_search_configs,
            kv_cache=kv_cache,
            save_kv_cache_path=save_kv_cache_path)

        # Set the search_config._max_seqlen
        for idx, request in enumerate(request_uids.view(-1).tolist()):
            init_seqlen = len(input_ids[idx]) - new_seq_batcher.offsets[idx]
            if kv_cache:
                init_seqlen += kv_cache[0][0].shape[-2]
            # TODO: change search_configs dict to list
            new_seq_batcher.search_configs[request]._max_seqlen = new_seq_batcher.search_configs[
                                                                      request].max_new_seqlen + init_seqlen

        # Merge
        if not self.seq_batchers[seq_batcher_cls]:
            self.seq_batchers[seq_batcher_cls].append(new_seq_batcher)
        else:
            self.seq_batchers[seq_batcher_cls][0].add_batch(new_seq_batcher)

        # collect the input into result
        for request_uid, output_id in zip(
                request_uids.view(-1).tolist(), output_ids):
            self.results[request_uid] = output_id

    def is_empty(self):
        return all(seq_batcher.is_empty()
                   for seq_batcher_list in self.seq_batchers.values()
                   for seq_batcher in seq_batcher_list)

    def total_seq_batcher_num(self):
        # This is provided to the consumers, used as part of the max_seq_batcher thresholding mechanism.
        return sum(
            len(seq_batcher_list)
            for seq_batcher_list in self.seq_batchers.values())

    def total_batch_size(self) -> Dict[Type[SeqBatcher], int]:
        # This is provided to the consumers, used as part of the max_batch_size thresholding mechanism.
        batch_size = {}
        for key, seq_batcher_list in self.seq_batchers.items():
            batch_size[key] = sum(seq_batcher.batch_size
                                  for seq_batcher in seq_batcher_list)
        return batch_size

    def inference_call(self) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        A sweep of inference calls on all seq_batchers in the scheduler
        Returns:
            output_ids (`List[List[int]`):
                About List[List[int]] structure, the outermost List[] corresponds to request_uid: List[int]. The
                inner List[int] is used to extend the past_output_ids: past_output_ids.extend(List[
                int]). This is the same form as the output from `add_request`.
            request_uids (`List[int]`):
                The request_uids that correspond to output_ids. Ordering may be different from input since
                batch_merge or batch_trim operation.
            exist_request_uids (`List[int]`):
                List[int] a list of request_uids that have finished.
        """

        output: List[List[int]] = []
        request_uids: List[int] = []
        exit_request_uids: List[int] = []
        for seq_batcher_cls in self.seq_batchers:
            seq_batcher_list_new = []
            for seq_batcher in self.seq_batchers[seq_batcher_cls]:
                output += seq_batcher.forward()
                request_uids += seq_batcher.request_uids.view(-1).tolist()

                exit_request_uids += seq_batcher.collect_and_trim()
                if not seq_batcher.is_empty():
                    seq_batcher_list_new.append(seq_batcher)

            self.seq_batchers[seq_batcher_cls] = seq_batcher_list_new

        return output, request_uids, exit_request_uids

    def increment_forward(self, count: int):
        # This serves as a demo of how to use this scheduler
        # -> Dict[Type[SeqBatcher]: List[List[int]]]
        i = 0
        while i < count and not self.is_empty():
            output_ids, request_uids, _ = self.inference_call()

            # collect output
            for request_uid, output_id in zip(request_uids, output_ids):
                self.results[request_uid].extend(output_id)

            i += 1
            yield output_ids

    def collect_results(self):
        output = self.results
        self.results = defaultdict(list)
        return output

    def seq_batcher_split(self, seq_batcher_cls: Type[SeqBatcher],
                          seq_batcher_idx: int, partitions: List[List[int]]):
        seq_batcher = self.seq_batchers[seq_batcher_cls].pop(seq_batcher_idx)
        self.seq_batchers[seq_batcher_cls].extend(
            seq_batcher.split(partitions))
