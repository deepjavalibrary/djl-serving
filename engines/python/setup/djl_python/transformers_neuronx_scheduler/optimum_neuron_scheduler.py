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
# The below code is heavily inspired from Optimum Neuron under the following link:
# https://github.com/huggingface/optimum-neuron/blob/974f34336bb36b1b64890c191c558a1575372be7/text-generation-inference/server/text_generation_server/generator.py
import copy
from enum import Enum
from typing import List, Optional

import torch
import logging
from transformers import PreTrainedTokenizerBase
from transformers.generation import GenerationConfig

from djl_python.rolling_batch.rolling_batch import Request
from djl_python.transformers_neuronx_scheduler.token_selector import TokenSelector
from djl_python.transformers_neuronx_scheduler.utils import Generation, FinishReason, GeneratedText


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

    def assign(self, request: Request, generation_config: GenerationConfig):
        """Assign a request to a slot.

        Args:
            request (`Request`):
                The request to be assigned. Contains the inputs and tokens selection parameters.
            generation_config (`transformers.GenerationConfig`):
                The base generation config (might be modified by the request generation parameters).
        """
        self._state = Slot.State.READY
        self._request_id = request.id
        self._inputs = request.input_text
        self._generation_config = copy.deepcopy(generation_config)
        # Update generation config with token chooser parameters
        param = request.parameters
        self._generation_config.temperature = param.get("temperature", 0.9)
        self._generation_config.top_k = param.get("top_k", 0)
        self._generation_config.top_p = param.get("top_p", 1.0)
        self._generation_config.typical_p = param.get("typical_p", 1.0)
        self._generation_config.do_sample = param.get("do_sample", False)
        self._generation_config.repetition_penalty = param.get(
            "repetition_penalty", 1.0)
        # TODO: seed, watermark
        self._generation_config.max_new_tokens = param.get(
            "max_new_tokens", 30)
        # TODO: stop_sequences, ignore_eos_token

    def reset(self, input_ids, attention_mask, selector):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`torch.LongTensor`):
                The new input_ids to use to generate the next token.
            attention_mask: (`torch.LongTensor`):
                The new attention_mask to use to generate the next token.
            selector: (`optimum.neuron.generation.TokenSelector`):
                An object implementing the updated token selection logic.
        """
        self._tokens = input_ids
        self._mask = attention_mask
        self._selector = selector

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

    def select(self, input_ids: torch.LongTensor,
               logits: torch.Tensor) -> torch.LongTensor:
        """Select the next token from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A scalar torch.LongTensor` containing the selected token.
        """
        return self._selector.select(input_ids, logits)[0]

    @property
    def stopped(self) -> bool:
        return self._selector.stopping_criteria(self._tokens, None)

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


class NeuronGenerator:
    """A Generator for Neuron models."""

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, batch_size,
                 n_positions):
        self.model = model
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.special_tokens = [
            self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        ]
        self.slots = [Slot(i) for i in range(batch_size)]
        self.batch_size = batch_size
        self.n_positions = n_positions

    def prefill(self, new_requests: List[Request]):
        """Prefill new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        active_slots = slots[Slot.State.READY]
        empty_slots = slots[Slot.State.EMPTY]
        if len(empty_slots) < len(new_requests):
            raise ValueError(
                f"Cannot prefill {len(new_requests)} new request(s) with only {len(empty_slots)} empty slots."
                f"Please align the number of concurrent requests with the static batch size: {self.batch_size}."
            )
        # Assign each request to an empty slot
        logging.debug(
            f"Prefilling {len(new_requests)} new request(s) with {len(empty_slots)} empty slot(s)"
        )
        for request in new_requests:
            slot = empty_slots.pop()
            slot.assign(request, self.model.generation_config)
            logging.debug(
                f"Request {slot.request_id} assigned to slot {slot.id}")
        # Reconstruct the full inputs (without padding)
        inputs = [slot.inputs for slot in self.slots]
        # Tokenize with padding
        padded_inputs = self.tokenizer(inputs,
                                       return_tensors="pt",
                                       padding=True)
        #  If needed truncate sequences to fit into the static dimensions
        seq_length = min(padded_inputs.input_ids.shape[-1], self.n_positions)
        input_ids = padded_inputs.input_ids[:, :seq_length]
        attention_mask = padded_inputs.attention_mask[:, :seq_length]
        # Each slot must be reset with the padded inputs
        for i, slot in enumerate(self.slots):
            if slot.state != slot.state.EMPTY:
                slot_input_ids = input_ids[i:i + 1, :]
                # Padded input ids are also required to set logits processors and stopping criterias
                selector = TokenSelector.create(slot_input_ids,
                                                slot.generation_config,
                                                self.model, self.n_positions)
                slot_input_ids = slot_input_ids.squeeze().type(torch.int64)
                slot_attention_mask = attention_mask[i]
                slot.reset(slot_input_ids, slot_attention_mask, selector)
        # Clear KV cache
        self.model.reset_generation()
        # Pause previously active slots during generation.
        # Their KV cache will be prefilled but new tokens will be ignored, as they
        # have already been generated and sent back in the last decode.
        for slot in active_slots:
            slot.pause()
        generation = self._generate_token(input_ids, attention_mask)
        # Reactivate previously active slots for the next decode.
        for slot in active_slots:
            slot.resume()
        logging.debug("Model ready for decoding")
        return generation

    def decode(self) -> List[Generation]:
        """Decode the specified prefilled requests.

        Args:
            batches (`List[CachedBatch]`):
                A list of previous batches containing the prefilled requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        # Reconstruct input_ids and attention_mask from slots
        input_ids = None
        attention_mask = None
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.EMPTY:
                if input_ids is None:
                    # Create blank inputs covering all slots (even empty ones)
                    input_ids = torch.full(
                        [self.model.batch_size, 1],
                        fill_value=self.tokenizer.eos_token_id,
                        dtype=torch.int64)
                # input_ids are simply the tokens generated by the last decode or prefill requests (other tokens are cached)
                input_ids[i, 0] = slot.next_token
                if attention_mask is None:
                    # Create default mask covering all slots (even empty ones)
                    attention_mask = torch.zeros(
                        [self.model.batch_size,
                         slot.attention_mask.size(-1)],
                        dtype=torch.int64)
                    attention_mask[:, -1] = 1
                attention_mask[i, :] = slot.attention_mask
        if input_ids is None:
            raise ValueError(
                "Unable to decode tokens for non-prefilled batches (probably due to a previous failure)"
            )
        return self._generate_token(input_ids, attention_mask)

    def _generate_token(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None
    ) -> List[Generation]:
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids, attention_mask)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
        )
        generations = []
        request_ids = []
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.READY:
                continue
            request_id = slot.request_id
            request_ids.append(request_id)
            next_token_logits = outputs.logits[i:i + 1, -1, :]
            slot_input_ids = input_ids[i:i + 1, :]
            next_token = slot.select(slot_input_ids, next_token_logits)
            next_token_text = self.tokenizer.decode(next_token)
            if not slot.generated_text.endswith(
                    " ") and not next_token_text.startswith(" "):
                # Some tokenizers do not prepend spaces automatically when decoding a single token
                contextual_text = self.tokenizer.decode(
                    [slot.next_token, next_token])
                if contextual_text[:-len(next_token_text)].endswith(" "):
                    next_token_text = " " + next_token_text
            slot.append(next_token, next_token_text)
            generated_text = None
            finish_reason = None
            if next_token == self.tokenizer.eos_token_id:
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif slot.stopped:
                finish_reason = FinishReason.FINISH_REASON_STOP_SEQUENCE
            if finish_reason is not None:
                # We must include the generated text for each finished sequence in the response
                generated_text = GeneratedText(
                    text=slot.generated_text,
                    generated_tokens=slot.generated_tokens,
                    finish_reason=finish_reason,
                    seed=0)
                logging.debug(
                    f"Finished generating tokens for request {request_id}")
                # mark the slot as available
                slot.clear()
            generations.append(
                Generation(
                    request_id=request_id,
                    prefill_tokens=None,
                    token_id=next_token,
                    token_logprob=None,
                    token_text=next_token_text,
                    token_is_special=(next_token in [self.special_tokens]),
                    generated_text=generated_text,
                ))
        return generations

    def filter(self, request_ids: List[int]):
        """Remove requests that are not listed from the specified batch

        Args:
            batch_id (`int`):
                The id of a cached batch.
            request_ids(`List[int]`):
                The list of requests that must be kept.

        Return:
            A `CachedBatch` containing the pending requests.
        """
        self._clear(request_ids)

    def clear(self):
        """Remove all requests from the generator"""
        return self._clear([])

    def _clear(self, request_ids: List):
        for slot in self.slots:
            if slot.state != Slot.State.EMPTY and slot.request_id not in request_ids:
                logging.debug(f"Removing request {slot.request_id}")
                slot.clear()
