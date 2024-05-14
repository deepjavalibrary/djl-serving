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

import copy
import torch
from enum import Enum
from typing import Optional, Any, List
from transformers.generation import GenerationConfig
from djl_python.rolling_batch.rolling_batch import Request, filter_unused_generation_params
from djl_python.transformers_neuronx_scheduler.utils import Generation, FinishReason, GeneratedText, TokenDecoder

GENERATION_PARAMS = list(GenerationConfig().__dict__.keys())
TOKEN_SELECTION_PARAMS = ["seed", "ignore_eos", "stop_token_ids"]
NEURON_GENERATION_PARAMS = set(GENERATION_PARAMS + TOKEN_SELECTION_PARAMS)


def translate_neuronx_params(parameters: dict) -> dict:
    # TODO: Remove this once presence_penalty is supported
    if "presence_penalty" in parameters.keys(
    ) and "repetition_penalty" not in parameters.keys():
        parameters["repetition_penalty"] = float(
            parameters.pop("presence_penalty")) + 2.0
    return parameters


class Slot:
    """Represents a slot in a static batch"""

    class State(Enum):
        EMPTY = 0
        PAUSE = 1
        READY = 2

    def __init__(self, id: int):
        self._id = id
        self.clear()

    def clear(self):
        """Clear the slot and mark it as available."""
        self._state = Slot.State.EMPTY
        self._request_id = None
        self._inputs = ""
        self._generation_config = None
        self._tokens = []
        self._mask = []
        self._selector = None
        self._generated_tokens = 0
        self._next_token_text = ""
        self._cache_id = torch.zeros(1)
        self._token_decoder = None
        self._token_acceptor = None
        self._special_tokens = []
        self._ignore_eos_id = False
        self.seed = 0

    @property
    def id(self) -> int:
        return self._id

    @property
    def state(self) -> "Slot.State":
        return self._state

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def inputs(self) -> str:
        return self._inputs

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def generated_tokens(self) -> torch.LongTensor:
        return self._generated_tokens

    @property
    def decoder(self) -> TokenDecoder:
        return self._token_decoder

    @property
    def acceptor(self) -> Optional[Any]:
        return self._token_acceptor

    @property
    def special_tokens(self) -> List[int]:
        return self._special_tokens

    def build_eos_token_ids(self, params) -> List[int]:
        if isinstance(self._generation_config.eos_token_id, int):
            eos_token_ids = [self._generation_config.eos_token_id]
        else:
            eos_token_ids = [*self._generation_config.eos_token_id]

        stop_token_ids = params.get("stop_token_ids")
        if isinstance(stop_token_ids, int):
            eos_token_ids.append(stop_token_ids)
        elif stop_token_ids:
            eos_token_ids = eos_token_ids + stop_token_ids
        return eos_token_ids

    def assign(self, request: Request, generation_config: GenerationConfig,
               tokenizer, acceptor):
        """Assign a request to a slot.

        Args:
            request (`Request`):
                The request to be assigned. Contains the inputs and tokens selection parameters.
            generation_config (`transformers.GenerationConfig`):
                The base generation config (might be modified by the request generation parameters).
            tokenizer:
                The tokenizer used to decode token.
            acceptor:
                The speculative token acceptor available when speculative decoding
        """
        self._state = Slot.State.READY
        self._request_id = request.id
        self._inputs = request.input_text
        self._generation_config = copy.deepcopy(generation_config)
        # Update generation config with token chooser parameters
        param = translate_neuronx_params(request.parameters)
        self.seed = 0
        self._token_acceptor = acceptor
        self._generation_config.do_sample = param.get("do_sample", False)
        if self._generation_config.do_sample:
            self._generation_config.temperature = param.get("temperature", 0.9)
            self._generation_config.top_k = param.get("top_k", 0)
            self._generation_config.top_p = param.get("top_p", 1.0)
            self._generation_config.typical_p = param.get("typical_p", 1.0)
            self.seed = int(param.get("seed", 0))

        self._generation_config.repetition_penalty = param.get(
            "repetition_penalty", 1.0)
        self._generation_config.max_new_tokens = param.get(
            "max_new_tokens", 30)
        self._generation_config.eos_token_id = self.build_eos_token_ids(param)
        self._token_decoder = TokenDecoder(tokenizer)
        self._ignore_eos_id = param.pop("ignore_eos", False)
        filter_unused_generation_params(param,
                                        NEURON_GENERATION_PARAMS,
                                        "neuron",
                                        remove_unused_params=True)

    def reset(self, input_ids, attention_mask, selector, cache_id):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`torch.LongTensor`):
                The new input_ids to use to generate the next token.
            attention_mask: (`torch.LongTensor`):
                The new attention_mask to use to generate the next token.
            selector: (`optimum.neuron.generation.TokenSelector`):
                An object implementing the updated token selection logic.
            cache_id: (torch.LongTensor):
                The new cache_ids to use to generate the next token
        """
        self._tokens = input_ids
        self._mask = attention_mask
        self._selector = selector
        self._cache_id = cache_id

    def pause(self):
        """Mark the current slot as paused for generation.

        Note that the KV cache for this slot will still be filled.
        """
        self._state = Slot.State.PAUSE

    def resume(self):
        """Mark the slot as ready for generation."""
        if self._state == Slot.State.PAUSE and self.next_token is not None:
            # The generation of this slot was inhibited during a prefill, but it
            # already had a pending token, so we need to increase attention mask
            self._mask = torch.cat([self._mask, torch.LongTensor([1])])
        self._state = Slot.State.READY

    def append(self, next_token: int, next_token_text: str):
        """Append a new generated token to this slot

        The new token is added to the list of generated tokens, which impacts
        directly the generated_text and stopped property.

        The new token is however not added immediately to the slot inputs: it will
        be added later on when it has effectively been used to produce the next token.

        Args:
            next_token (`int`):
                The newly generated token.
            next_token_test (`str`):
                The corresponding decoded text.
        """
        self._tokens = torch.cat(
            [self._tokens, torch.LongTensor([next_token])])
        self._mask = torch.cat([self._mask, torch.LongTensor([1])])
        self._generated_tokens += 1
        # Now that a new token has been generated, we can append the previous one to the inputs
        self._inputs += self._next_token_text
        self._next_token_text = next_token_text
        self.increment_cache_id()

    def select(self, input_ids: torch.LongTensor,
               logits: torch.Tensor) -> (torch.LongTensor, torch.Tensor):
        """Select the next token from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A scalar torch.LongTensor` containing the selected token.
        """
        next_ids, next_log_probs = self._selector.select(input_ids, logits)
        return next_ids[0], next_log_probs

    def increment_cache_id(self):
        self._cache_id += 1

    def trim_cache_id(self):
        self._cache_id = self._cache_id.max()

    def is_slot_eos_token(self, token) -> bool:
        if self._ignore_eos_id:
            return False
        if hasattr(self._generation_config, "eos_token_id"):
            if isinstance(self._generation_config.eos_token_id, int):
                return token == self._generation_config.eos_token_id
            else:
                return token in self._generation_config.eos_token_id
        else:
            return False

    def accept_speculated_tokens(self, *args, **kwargs):
        return self._token_acceptor(*args, **kwargs)

    @property
    def stopped(self) -> bool:
        return self._selector.stopping_criteria(self._tokens, None)

    @property
    def tokens(self) -> torch.LongTensor:
        return self._tokens

    @property
    def generated_text(self) -> str:
        return self._inputs + self._next_token_text

    @property
    def next_token(self) -> int:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    @property
    def attention_mask(self) -> torch.LongTensor:
        return self._mask

    @property
    def max_token(self) -> int:
        return self._generation_config.max_length

    @property
    def cache_id(self) -> torch.LongTensor:
        return self._cache_id
