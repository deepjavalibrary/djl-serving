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
# The below code is heavily inspired from Optimum Neuron under the following link:
# https://github.com/huggingface/optimum-neuron/blob/main/optimum/neuron/generation/token_selector.py

import copy
import logging
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerationMode

from optimum.neuron.generation import TokenSelector

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# TODO: This is a temporary solution to avoid Optimum's dependency on transformers<4.42.
class OptimumTokenSelector(TokenSelector):
    """Implements the token selection logic corresponding to a generation configuration.

    This class combines and uses the logits processors and stopping criterias implemented in
    the transformers library.

    The algorithm to select these objects is heavily inspired by the transformers `GenerationMixin.generate()`
    method, but the actual token selection methods are specific.

    The reason why this class does not inherit from `GenerationMixin` is because it does not
    include the code to produce the tokens logits.
    Separating the production of the tokens logits from the tokens selection allows this class
    to be used with different generation paradigms, either synchronously using a single `TokenSelector` in
    `GenerationMixin.generate()` or asynchronously using multiple `TokenSelector` inside an inference endpoint.

    The constructor of this class should not be called directly: instances should be obtained by
    calling `TokenSelector.create()`.
    """

    def select_with_logprobs(
            self, input_ids: torch.LongTensor,
            logits: torch.Tensor) -> (torch.LongTensor, torch.Tensor):
        """Select the next tokens from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the selected tokens.
            `torch.Tensor`: A `torch.Tensor` containing the selected logprobs.
        """
        # Cast to int64 for Repetition Penalty logit processor support
        scores = self.logits_processor(input_ids.to(torch.int64), logits)
        logprobs = torch.log_softmax(scores, -1)
        if self.mode == GenerationMode.SAMPLE:
            next_ids = self._sample(scores)
        else:
            next_ids = torch.argmax(scores, dim=-1)
        next_logprobs = torch.gather(logprobs, 1, next_ids.view(-1,
                                                                1)).view(-1)
        return next_ids, next_logprobs
