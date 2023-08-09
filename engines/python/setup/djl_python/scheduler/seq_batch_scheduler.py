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
from collections import defaultdict, OrderedDict
from typing import Union, Tuple, List, Dict, Type

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.lm_block import LMBlock
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.seq_batcher_impl import GreedySeqBatcher, ContrastiveSeqBatcher
from djl_python.scheduler.utils import compute_kv_cache

SEARCH_ALGORITHM_TO_CLASS = {
    "greedy": GreedySeqBatcher,
    "sampling": GreedySeqBatcher,
    "contrastive": ContrastiveSeqBatcher
}


class SeqBatchScheduler:
    """
    This is a scheduler that manages the SeqBatcher, providing API which allows for actions like addBatch,
    collectResults.
    """

    def __init__(self, lm_block: LMBlock, default_search_algorithm: str,
                 default_config: SearchConfig):
        self.default_search_configs = defaultdict(lambda: default_config)
        self.default_seq_batcher_cls = SEARCH_ALGORITHM_TO_CLASS[
            default_search_algorithm]
        self.lm_block = lm_block
        self.results: Dict[int, List[int]] = defaultdict(list)

        self.seq_batchers: Dict[
            Type[SeqBatcher]:List[SeqBatcher]] = defaultdict(list)

        self.lru_kv_cache = OrderedDict()
        self.lru_max_size = 10

    def add_request(self,
                    input_ids: torch.Tensor,
                    request_uids: torch.Tensor,
                    search_algorithm: str = None,
                    search_configs: List[SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None,
                    kv_cache_prompt_ids: Union[Dict[int, torch.tensor],
                                               None] = None):
        """
        Args: kv_cache_prompt_ids = {request_uid -> List[token_ids]}
        """

        # Find the requests that uses kv_cache_prompt_ids
        index_not_use_prompt = []
        search_configs_not_use_prompt = []
        if search_configs:
            for idx, search_config in enumerate(search_configs):
                if search_config.use_lru_kv_cache:
                    request_uid = request_uids[idx].item()
                    if request_uid not in kv_cache_prompt_ids:
                        raise Exception(
                            f"request_uids = {request_uid}: search_config says use_kv_cache_prompt, "
                            f"but the prompt_ids is not provided.")
                    prompt_ids_tensor = kv_cache_prompt_ids[request_uid]
                    key = tuple(prompt_ids_tensor.flatten().tolist())
                    # lru operations
                    if key not in self.lru_kv_cache:
                        if len(self.lru_kv_cache) + 1 > self.lru_max_size:
                            # If cache size exceeds the maximum, remove by FIFO order
                            self.lru_kv_cache.popitem(last=False)
                        kv_cache_tuple = compute_kv_cache(
                            input_ids=prompt_ids_tensor,
                            lm_block=self.lm_block,
                            search_configs=[search_config])
                        kv_cache_new = []
                        for k, v in kv_cache_tuple:
                            k_new = k.cpu()
                            v_new = v.cpu()
                            kv_cache_new.append((k_new, v_new))
                        self.lru_kv_cache[key] = tuple(kv_cache_new)
                        self.lru_kv_cache.move_to_end(key)

                        # _add_request
                        self._add_request(input_ids[idx].view(1, -1),
                                          request_uids[idx].view(1, -1),
                                          search_algorithm, [search_config],
                                          kv_cache=kv_cache_tuple)
                    else:
                        # _add_request
                        self._add_request(input_ids[idx].view(1, -1),
                                          request_uids[idx].view(1, -1),
                                          search_algorithm, [search_config],
                                          kv_cache=self.lru_kv_cache[key])
                        self.lru_kv_cache.move_to_end(key)
                else:
                    index_not_use_prompt.append(idx)
                    search_configs_not_use_prompt.append(search_config)
        else:
            index_not_use_prompt = list(range(input_ids.shape[0]))
            search_configs_not_use_prompt = None

        if index_not_use_prompt:
            index_not_use_prompt = torch.tensor(index_not_use_prompt)
            self._add_request(input_ids[index_not_use_prompt],
                              request_uids[index_not_use_prompt],
                              search_algorithm, search_configs_not_use_prompt,
                              kv_cache)

    def _add_request(self,
                     input_ids: torch.Tensor,
                     request_uids: torch.Tensor,
                     search_algorithm: str = None,
                     search_configs: List[SearchConfig] = None,
                     kv_cache: Union[Tuple, None] = None):
        # TODO: next, this will take an argument of `action`, computed by self.optimal_action.
        device = input_ids.device
        request_uids = request_uids.to(device)
        seq_batcher_cls = SEARCH_ALGORITHM_TO_CLASS.get(
            search_algorithm, self.default_seq_batcher_cls)
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

        # Corner case: input_ids are empty. Pad them.
        if input_ids.numel() == 0:
            batch_size = input_ids.shape[0]
            input_ids = torch.zeros(batch_size,
                                    1,
                                    dtype=torch.int64,
                                    device=input_ids.device)
            for i in range(batch_size):
                input_ids[i, 0] = self.default_search_configs[
                    request_uids[i].item()].pad_token_id

        # Prefill
        new_seq_batcher, output_ids = seq_batcher_cls.init_forward(
            input_ids=input_ids,
            request_uids=request_uids,
            lm_block=self.lm_block,
            search_configs=self.default_search_configs,
            kv_cache=kv_cache)

        # Set the search_config._max_seqlen
        for idx, request in enumerate(request_uids.view(-1).tolist()):
            init_seqlen = len(
                input_ids[idx]) - new_seq_batcher.offsets[idx].item()
            if kv_cache:
                init_seqlen += kv_cache[0][0].shape[-2]
            # TODO: change search_configs dict to list
            new_seq_batcher.search_configs[
                request]._max_seqlen = new_seq_batcher.search_configs[
                    request].max_new_seqlen + init_seqlen

        # Merge
        # TODO: next, an optimal action needs to be first computed, according to which the merge is done.
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

    def optimal_action(self,
                       input_ids: torch.Tensor,
                       request_uids: torch.Tensor,
                       seq_batcher_cls: Type[SeqBatcher] = None,
                       search_configs: List[SearchConfig] = None,
                       kv_cache: Union[Tuple, None] = None,
                       save_kv_cache_path: str = None):
        """
        Get the optimal merging action computed according to the added request and the current scheduler status.

        Args:
            The request information.

        Return:
            Optimal merging action: `Action`:
                1. choose a seq_batcher to merge in
                2. split a seq_batcher
                3. rearrange the whole seq_batcher list
        """

        # This is provided to the consumers to be used as part of the max_seq_batcher thresholding mechanism.
        pass

    @staticmethod
    def optimal_partition(seq_length_list: List[int],
                          num_part: int) -> Tuple[int, List[List[int]]]:
        """
        total_padding, opt_partition = self.optimal_partition(
            seq_length_list, num_part)

        Args:
            seq_length_list: list of sequence lengths. Sorted in descending order.
            num_part: number of parts in a partition

        Return:
            cost: total padding
            opt_partition (`List[List[int]]`): optimal partition stored as List of List of sequence index
        """
        if num_part <= 0:
            raise Exception("Illegal argument.")

        batch_size = len(seq_length_list)
        arr = seq_length_list

        # dp[i][k] stores the optimal cost of partition the suffix array arr[i:] into k parts.
        dp = [[-1 for _ in range(num_part + 1)] for _ in range(batch_size)]

        # dp_parts[i][k] stores the corresponding optimal partition. dict: (i, k) -> List[int]
        dp_parts = defaultdict(list)

        def dp_recur(idx, k) -> Tuple[int, List[int]]:
            """
            dp(idx, k) returns the optimal cost of partition the suffix array arr[i:] into k parts.
            """
            if k == 1:
                if idx == batch_size:
                    return 0, []
                if dp[idx][k] > -1:
                    return dp[idx][k], dp_parts[idx, k]
                else:
                    max_seq_size = arr[idx]
                    dp[idx][k], dp_parts[idx, k] = sum(
                        max_seq_size - arr[i]
                        for i in range(idx, batch_size)), [idx]
                    return dp[idx][k], dp_parts[idx, k]

            if idx == batch_size:
                return 0, []

            if dp[idx][k] > -1:
                return dp[idx][k], dp_parts[idx, k]

            padding_leftmost_part = 0
            opt_cost, opt_cuts = float('inf'), None
            for i in range(idx, batch_size):
                padding_leftmost_part += arr[idx] - arr[i]
                padding_suffix_part, opt_cuts_suffix_part = dp_recur(
                    i + 1, k - 1)
                if padding_leftmost_part + padding_suffix_part < opt_cost:
                    opt_cost = padding_leftmost_part + padding_suffix_part
                    opt_cuts = [i + 1] + opt_cuts_suffix_part

            dp[idx][k], dp_parts[idx, k] = opt_cost, opt_cuts
            return opt_cost, opt_cuts

        optimal_cost, optimal_cuts = dp_recur(0, num_part)

        # Convert the cuts to parts of sequence index list
        optimal_part = []
        for i in range(len(optimal_cuts)):
            optimal_part.append(
                list(
                    range(0 if i == 0 else optimal_cuts[i - 1],
                          optimal_cuts[i])))
        optimal_part.append(
            list(
                range(optimal_cuts[-1] if len(optimal_cuts) > 0 else 0,
                      batch_size)))
        return optimal_cost, optimal_part

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
        """
        Split a seq_batcher in the seq_batcher_list located at seq_batcher_idx, into parts according to `partition`.
        Args:
            seq_batcher_cls: SeqBatcher type
            seq_batcher_idx: idx in the seq_batcher_list
            partitions: contains the seq_batcher_idx partitioned into lists.
        """

        seq_batcher = self.seq_batchers[seq_batcher_cls].pop(seq_batcher_idx)
        self.seq_batchers[seq_batcher_cls].extend(
            seq_batcher.split(partitions))

    def get_request_ids(self):
        request_uids = []
        for seq_batcher_cls in self.seq_batchers:
            for seq_batcher in self.seq_batchers[seq_batcher_cls]:
                request_uids += seq_batcher.request_uids.view(-1).tolist()

        return request_uids
