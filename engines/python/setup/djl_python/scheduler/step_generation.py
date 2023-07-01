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
import bisect
from collections import defaultdict

import torch
from torch.nn.functional import normalize, softmax
from typing import Tuple, List, Dict
from djl_python.scheduler.search_config import SearchConfig
import numpy, heapq


def contrastive_step_generate(top_k_ids: torch.Tensor,
                              top_k_probs: torch.Tensor,
                              top_k_hidden_states: torch.Tensor,
                              context_hidden_states: torch.Tensor,
                              offsets: torch.Tensor, alpha: float):
    # top_k_ids: [batch, topK]
    # top_k_probs:  [batch, topK]
    # top_k_hidden_states: [batch*topK, seq=1, dim]
    # context_hidden_states: [batch, past_seq, dim]
    # offsets: [batch, 1]

    batch_size, topk = top_k_ids.shape
    hidden_dim = top_k_hidden_states.shape[-1]

    # [batch*topK, seq=1, dim] -> [batch, topK, dim]
    top_k_hidden_states = top_k_hidden_states.view(batch_size, topk,
                                                   hidden_dim)

    # [batch, topK, dim] * [batch, past_seq, dim] -> [batch, topK, past_seq]
    top_k_hidden_states = normalize(top_k_hidden_states, p=2, dim=2)
    context_hidden_states = normalize(context_hidden_states, p=2, dim=2)

    cos_similarity = torch.bmm(top_k_hidden_states,
                               context_hidden_states.permute(0, 2, 1))

    for i in range(offsets.numel()):
        cos_similarity[i, :, :offsets[i]] = -1

    # [batch, topK, past_seq] -> [batch, topK]
    top_k_score_part1 = torch.max(cos_similarity, dim=2).values
    top_k_score_part2 = top_k_probs

    top_k_score = top_k_score_part2.mul_(1 - alpha).sub_(
        top_k_score_part1.mul_(alpha))

    # [batch, topK] => [batch, 1]
    select = torch.argmax(top_k_score, dim=1).view(-1)
    a_range = torch.arange(top_k_ids.shape[0])
    output_ids = top_k_ids[a_range, select, ...].view(-1, 1)
    return output_ids, select


def sampling_step_generate(logits: torch.tensor,
                           search_configs: List[SearchConfig],
                           sampler_bucket_sort_cache=None):
    """
    Greedy, topK, topP

    Args:
        logits: [batch, vocab_size]
        search_configs: [batch]
        sampler_bucket_sort_cache: Tuple[collector, k_config_list, p_config_list].
            This is cached output of sampler_bucket_sort result used through inferences.

    Return:
        token_id: [batch, 1]
    """
    collector, k_config_list, tmprtr_list_for_k, p_config_list, tmprtr_list_for_p = sampler_bucket_sort(
        search_configs
    ) if not sampler_bucket_sort_cache else sampler_bucket_sort_cache

    output_ids_greedy = greedy_step_generate(logits[collector['greedy'], :])
    output_ids_topk = topk_step_generate(logits[collector['topk'], :],
                                         k_config_list, tmprtr_list_for_k)
    output_ids_topp = topp_step_generate(logits[collector['topp'], :],
                                         p_config_list, tmprtr_list_for_p)
    output_ids = torch.empty(len(search_configs),
                             dtype=torch.int64,
                             device=logits.device)
    output_ids[collector['greedy']] = output_ids_greedy.view(-1)
    output_ids[collector['topk']] = output_ids_topk.view(-1)
    output_ids[collector['topp']] = output_ids_topp.view(-1)

    return output_ids.view(-1, 1)


def sampler_bucket_sort(search_configs: List[SearchConfig]):
    """
    Return:
        collector: Dict[str: List[int]],
        k_config_list: List[int],
        p_config_list: List[float]
    """

    collector = defaultdict(list)
    k_config_list = []
    p_config_list = []
    tmprtr_list_for_p = []
    tmprtr_list_for_k = []
    for idx, search_config in enumerate(search_configs):
        if not search_config.sampling:
            collector['greedy'].append(idx)
        elif search_config.topk > 0:
            collector['topk'].append(idx)
            k_config_list.append(search_config.topk)
            tmprtr_list_for_k.append(search_config.temperature)
        else:
            collector['topp'].append(idx)
            p_config_list.append(search_config.topp)
            tmprtr_list_for_p.append(search_config.temperature)
    return collector, k_config_list, tmprtr_list_for_k, p_config_list, tmprtr_list_for_p


def greedy_step_generate(logits: torch.Tensor, k: int = 1) -> torch.tensor:
    """
    Args:
        logits: [batch, vocab_size].
            This logits can also be probability inputs, since probs = softmax(logits, dim=1) .

    Return:
        indices: [batch, k]
    """
    return torch.topk(logits, k=k, dim=-1, largest=True, sorted=False).indices


def topk_step_generate(logits, k_config_list: List[int],
                       tmprtr_list_for_k: List[float]):
    """
    Returns the token ids of the top k selection. If logits is tensor([]), the output should be tensor([]) too.
    """
    if logits.numel() == 0:
        return torch.tensor([], dtype=torch.int64, device=logits.device)

    batch_size, vocab_size = logits.size()

    # result
    indices = numpy.empty(batch_size, dtype=numpy.int64)

    # Find the candidate: O(k * log(vocab_size))
    for i in range(batch_size):
        k = k_config_list[i]
        topk_values, topk_indices = torch.topk(logits[i],
                                               k=k,
                                               dim=-1,
                                               largest=True,
                                               sorted=True)
        # At this step the truncated prob is normalized
        probs = softmax(topk_values / tmprtr_list_for_k[i], dim=-1)

        indices[i] = topk_indices[torch.multinomial(probs, 1)]

    return torch.from_numpy(indices).view(-1, 1)


def topp_step_generate(logits, p_config_list: List[float],
                       tmprtr_list_for_p: List[float]):
    """
    Returns the token ids of the top p selection. If logits is tensor([]), the output should be tensor([]) too.

    Args:
        logits: [batch, vocab_size].

    Return:
        indices: [batch, 1]
    """
    if logits.numel() == 0:
        return torch.tensor([], dtype=torch.int64, device=logits.device)

    batch_size, vocab_size = logits.size()

    # Apply temperature to logits
    temperature = torch.tensor(tmprtr_list_for_p,
                               device=logits.device).view(-1, 1)
    logits = logits / temperature

    # Apply softmax to obtain probabilities
    probabilities = softmax(logits, dim=-1)

    # random number
    random_array = numpy.random.rand(batch_size)

    # result
    indices = numpy.empty(batch_size, dtype=numpy.int64)

    for i in range(batch_size):
        cum_prob = 0
        probs = [(-probabilities[i, j].item(), j)
                 for j in range(vocab_size)]  # O(vocab_size)
        heapq.heapify(probs)  # O(vocab_size)

        # Find the candidates: O(k * log(vocab_size))
        candidate_cum_probs = []
        candidate_ids = []
        while probs:
            neg_prob, index = heapq.heappop(probs)
            cum_prob -= neg_prob
            if cum_prob < p_config_list[i]:
                candidate_cum_probs.append(cum_prob)
                candidate_ids.append(index)
            else:
                candidate_cum_probs.append(cum_prob)
                candidate_ids.append(index)
                break

        # Renormalize and randomly select according to random_array
        rand_number = random_array[i].item()
        normalization_factor = candidate_cum_probs[-1]
        # Find the smallest idx whose cum_prob > rand_number[0, 1]. Both idx=0 and -1 are accessible.
        idx = bisect.bisect_right(
            [prob / normalization_factor for prob in candidate_cum_probs],
            rand_number)
        indices[i] = candidate_ids[idx]

    return torch.from_numpy(indices).view(-1, 1)


def beam_step_generate(last_probs: torch.Tensor, logits: torch.Tensor,
                       batch_len: int, beam_len: int):
    all_probs = torch.softmax(logits[:, -1, :],
                              dim=1).reshape(batch_len, beam_len, -1)
    top_k = torch.topk(all_probs,
                       k=beam_len,
                       dim=-1,
                       largest=True,
                       sorted=False)
    output_ids = top_k[1]
    step_probs = top_k[0]

    # Chain the probability
    # [batch, beamSource] -> [batch, beamSource, 1]
    last_probs = last_probs.reshape(batch_len, beam_len, 1)
    # [batch, beamSource, beamChild]
    new_probs = torch.mul(step_probs, last_probs)

    topK = torch.topk(new_probs.reshape(batch_len, beam_len * beam_len),
                      k=beam_len,
                      dim=-1,
                      largest=True,
                      sorted=False)

    # The select indices act on (beamSource, beamChild) dimension. Decides how the new
    # generated tokenIds correspond to the past tokenIds.
    # [batch, beamNew].
    select = topK[1]
