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
        self.results: Dict[int, List[int]] = {}

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

        # prefill
        new_seq_batcher, output_ids = seq_batcher_cls.init_forward(
            input_ids=input_ids,
            request_uids=request_uids,
            lm_block=self.lm_block,
            search_configs=self.default_search_configs,
            kv_cache=kv_cache,
            save_kv_cache_path=save_kv_cache_path)

        # merge
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

    def inference_call(self):
        """
        A sweep of inference calls on all seq_batchers in the scheduler
        Returns:
            output_ids (`Dict[Type[SeqBatcher]: List[List[int]]]`):
                About List[List[int]] structure, the outermost List[] corresponds to request_uid: List[int]. The
                inner List[int] is used to extend the output_ids sequences: past_output_ids: List[int].extend(List[
                int]). This is the same form as the output from @add_request.
            exist_request_uids (`Dict[Type[SeqBatcher]: List[int]]`):
                List[int] a list of request_uids that have finished.
        """

        output: Dict[Type[SeqBatcher]:List[List[int]]] = defaultdict(list)
        exit_request_uids: Dict[Type[SeqBatcher]:List[int]] = defaultdict(list)
        for seq_batcher_cls in self.seq_batchers:
            seq_batcher_list_new = []
            for seq_batcher in self.seq_batchers[seq_batcher_cls]:
                output[seq_batcher_cls] += seq_batcher.forward()
                exit_request_uids[seq_batcher_cls].extend(
                    seq_batcher.collect_and_trim())

                if not seq_batcher.is_empty():
                    seq_batcher_list_new.append(seq_batcher)

            self.seq_batchers[seq_batcher_cls] = seq_batcher_list_new

        return output, exit_request_uids

    def increment_forward(self, count: int):
        # This serves as a demo of how to use this scheduler
        # -> Dict[Type[SeqBatcher]: List[List[int]]]
        i = 0
        while i < count and not self.is_empty():
            # Need to have a request_uids here before calling self.inference_call() since request_uids is modified
            # therein.
            request_uids: Dict[Type[SeqBatcher]:List[int]] = defaultdict(list)
            for seq_batcher_cls, seq_batcher_list in self.seq_batchers.items():
                for seq_batcher in seq_batcher_list:
                    request_uids[
                        seq_batcher_cls] += seq_batcher.request_uids.view(
                            -1).tolist()  # List[List[int]]

            output_ids, _ = self.inference_call()

            # collect output
            for request_uids_list, output_ids_list in zip(
                    request_uids.values(), output_ids.values()):
                for request_uid, output_id in zip(request_uids_list,
                                                  output_ids_list):
                    self.results[request_uid].extend(output_id)

            i += 1
            yield output_ids

    def collect_results(self):
        output = self.results
        self.results = {}
        return output

    def seq_batcher_split(self, seq_batcher_cls: Type[SeqBatcher],
                          seq_batcher_idx: int, partitions: List[List[int]]):
        seq_batcher = self.seq_batchers[seq_batcher_cls].pop(seq_batcher_idx)
        self.seq_batchers[seq_batcher_cls].extend(
            seq_batcher.split(partitions))
