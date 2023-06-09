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


def contrastive_step_generate(top_k_ids: torch.Tensor, logits: torch.Tensor,
                              context_hidden_states: torch.Tensor,
                              top_k_hidden_states: torch.Tensor,
                              offsets: torch.Tensor, alpha: float):
    # topKIds: [batch, topK]
    # attentionMask: [batch, past_seq]
    # logits:  [batch, vocabSize]
    # contextHiddenStates: [batch, past_seq, dim]
    # topkHiddenStates: [batch*topK, seq=1, dim]
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
        cos_similarity[i, :, :offsets[i].item()] = -1

    # [batch, topK, past_seq] -> [batch, topK]
    top_k_score_part1 = torch.max(cos_similarity, dim=2).values
    assert len(top_k_score_part1.shape) == 2
    # [batch, logitDim].gather([batch, topK) -> [batch, topK]
    top_k_score_part2 = torch.gather(softmax(logits, dim=1),
                                     dim=1,
                                     index=top_k_ids)

    top_k_score = torch.subtract(torch.mul(top_k_score_part2, 1 - alpha),
                                 torch.mul(top_k_score_part1, alpha))

    # [batch, topK] => [batch, 1]
    select = torch.argmax(top_k_score, dim=1).flatten()
    a_range = torch.arange(top_k_ids.shape[0])
    output_ids = top_k_ids[a_range, select, ...].view(-1, 1)
    return output_ids, select


def greedy_step_generate(logits: torch.Tensor):
    # logits: [batch, vocabSize]
    return torch.unsqueeze(torch.argmax(logits, dim=-1), dim=1)
