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
from torch.nn.functional import normalize, softmax
from typing import Tuple, List
from djl_python.scheduler.search_config import SearchConfig


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
    assert len(top_k_score_part1.shape) == 2
    top_k_score_part2 = top_k_probs

    top_k_score = top_k_score_part2.mul_(1 - alpha).sub_(
        top_k_score_part1.mul_(alpha))

    # [batch, topK] => [batch, 1]
    select = torch.argmax(top_k_score, dim=1).view(-1)
    a_range = torch.arange(top_k_ids.shape[0])
    output_ids = top_k_ids[a_range, select, ...].view(-1, 1)
    return output_ids, select


def greedy_step_generate(logits: torch.Tensor, k: int = 1):
    return torch.topk(logits, k=k, dim=-1, largest=True, sorted=False)


def sampling_step_generate(logits: torch.Tensor,
                           search_configs: List[SearchConfig]):
    pass


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
