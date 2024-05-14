#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
# The below code is heavily inspired from Transformers NeuronX under the following link:
# https://github.com/aws-neuron/transformers-neuronx/blob/main/src/transformers_neuronx/speculation.py

import torch
from typing import Optional, Tuple


class LMIDraftModelForSpeculation:
    """
    Standard Implementation of Draft model provider that auto-regressively speculates k tokens.
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def _context_block(self, input_ids,
                       start_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run context encoding network of the given model.

        Args:
            input_ids: The initial input tokens passed to the model
            start_ids: The offset from the beginning of each input in a batch.

        Returns:
            token: predicted next token
            score: predicted token score
        """
        next_token_scores = self.model(input_ids, None, start_ids)
        inputs = torch.argmax(next_token_scores, dim=1, keepdim=True)
        return inputs, next_token_scores

    def __call__(
        self,
        input_ids: torch.Tensor,
        k: int,
        attention_mask: Optional[torch.Tensor] = None,
        cache_ids: Optional[torch.Tensor] = None,
        start_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform standard autoregressive token generation using the draft model, to speculate k-tokens.

        Args:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seq_len)
            k: The number of speculative tokens
            cache_ids: The positions in the KV cache that should be updated. shape=(seq_len,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)

        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        start_len = 0
        if cache_ids:
            start_len = torch.min(cache_ids).item()

        if start_len == 0:  # run context network as cache_id location starts from 0.
            return self._context_block(input_ids, start_ids)

        next_token_scores = self.model(input_ids, cache_ids, start_ids)

        scores = []
        tokens = []

        # Speculate k tokens in auto regressive mode.
        for cur_len in range(start_len, start_len + k):
            next_len = cur_len + 1
            inputs = torch.argmax(next_token_scores, keepdim=True, dim=1)

            scores.append(next_token_scores)
            tokens.append(inputs)

            if next_len >= start_len + k:
                break

            cache_ids = torch.as_tensor([next_len], dtype=torch.int32)
            next_token_scores = self.model(inputs, cache_ids, start_ids)

        return torch.cat(tokens, dim=1), torch.cat(scores, dim=0)


class LMIGreedyTokenAcceptor:
    """
    Greedy implementation of a TokenAcceptor that only accepts the target models argmax
    """

    def __call__(
        self,
        draft_ids: torch.Tensor,
        draft_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len + 1 == target_token_len  # target should include additional token predicted

        target_probabilities = torch.softmax(target_scores, dim=-1)
        target_ids = torch.argmax(target_probabilities, dim=1)
        target_log_probs = torch.gather(torch.log_softmax(target_scores, -1),
                                        1, target_ids.view(-1, 1))
        draft_ids = draft_ids.squeeze()

        # Minimum will return the first occurrence of 0 or False (i.e. rejection)
        index = torch.where(draft_ids != target_ids[:-1])[0]

        if len(
                index
        ) == 0:  # If we didn't get a rejection this means all drafts were accepted
            return torch.unsqueeze(target_ids, 0), target_log_probs
        else:
            next_log_probs = target_log_probs[:index[0] + 1]
            return torch.unsqueeze(target_ids[:index[0] + 1],
                                   0), next_log_probs
