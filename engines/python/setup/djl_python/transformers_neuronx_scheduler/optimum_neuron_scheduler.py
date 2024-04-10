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

from typing import List, Optional, Dict

import torch
import logging
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizerBase

from djl_python.transformers_neuronx_scheduler.slot import Slot
from djl_python.rolling_batch.rolling_batch import Request, filter_unused_generation_params
from djl_python.transformers_neuronx_scheduler.token_selector import TokenSelector
from djl_python.transformers_neuronx_scheduler.utils import Generation, FinishReason, GeneratedText, TokenDecoder


class NeuronGenerator(ABC):
    """A Generator for Neuron models."""

    def __init__(self,
                 model,
                 tokenizer: PreTrainedTokenizerBase,
                 batch_size,
                 n_positions,
                 trim_cache=False):
        self.model = model
        self.tokenizer = tokenizer
        self.special_tokens = [
            self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        ]
        self.slots = [Slot(i) for i in range(batch_size)]
        self.batch_size = batch_size
        self.n_positions = n_positions
        self.trim_cache = trim_cache
        self.cache_ids = None

    def get_slots_by_state(self, slot_state: Slot.State):
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        return slots[slot_state]

    @abstractmethod
    def prefill(self, new_requests: List[Request]):
        """Prefill new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """

    @abstractmethod
    def decode(self) -> List[Generation]:
        """Decode the specified requests.

        Return:
            A list of `Generation` for each request.
        """

    @abstractmethod
    def prepare_model_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
    ) -> Dict:
        """Prepare inputs for batching strategy
        Args:
            input_ids (Tensor): input tokenized tensor values
            attention_mask (Tensor): mask for input tensor values
            cache_ids (Tensor): tensor indicating active sequences current cache location
            seq_ids (Tensor): tensor indicating which sequences are active

        Return:
            A dict of `Tensor`'s for each argument
        """

    def _generate_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
    ) -> List[Generation]:
        model_inputs = self.prepare_model_inputs(input_ids, attention_mask,
                                                 cache_ids, seq_ids)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
        )
        generations = []
        request_ids = []
        active_slots = self.get_slots_by_state(Slot.State.READY)
        for i, slot in enumerate(active_slots):
            request_id = slot.request_id
            request_ids.append(request_id)
            next_token_logits = outputs.logits[i:i + 1, -1, :]
            slot_input_ids = input_ids[i:i + 1, :]
            next_token, next_log_prob = slot.select(slot_input_ids,
                                                    next_token_logits)
            next_token_text = slot.decoder.decode(next_token.item())
            if self.trim_cache:
                slot.trim_cache_id()
            slot.append(next_token, next_token_text)
            generated_text = None
            finish_reason = None
            if slot.is_slot_eos_token(next_token):
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif next_token == self.tokenizer.eos_token_id:
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif slot.stopped:
                finish_reason = FinishReason.FINISH_REASON_LENGTH
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
                    token_logprob=next_log_prob,
                    token_text=next_token_text,
                    token_is_special=(next_token in [self.special_tokens])
                    or (finish_reason == FinishReason.FINISH_REASON_EOS_TOKEN),
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


class ContinuousBatchingNeuronGenerator(NeuronGenerator):
    """A Generator for Neuron models using TNXs continuous batching KV cache."""

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, batch_size,
                 n_positions):
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        super().__init__(model,
                         tokenizer,
                         batch_size,
                         n_positions,
                         trim_cache=True)

    def prefill(self, new_requests: List[Request]):
        """Prefill new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        active_slots = self.get_slots_by_state(Slot.State.READY)

        # pause the currently active slots, we will only run prefill on new slots
        for slot in active_slots:
            slot.pause()

        empty_slots = self.get_slots_by_state(Slot.State.EMPTY)
        if len(empty_slots) < len(new_requests):
            raise ValueError(
                f"Cannot prefill {len(new_requests)} new request(s) with only {len(empty_slots)} empty slots."
                f"Please align the number of concurrent requests with the static batch size: {self.batch_size}."
            )
        # Assign each request to an empty slot
        logging.debug(
            f"Prefilling {len(new_requests)} new request(s) with {len(empty_slots)} empty slot(s)"
        )

        prefill_slots = []
        for request in new_requests:
            slot = empty_slots.pop()
            slot.assign(request, self.model.generation_config, self.tokenizer)
            prefill_slots.append(slot)
            logging.debug(
                f"Request {slot.request_id} assigned to slot {slot.id}")

        inputs = [slot.inputs for slot in prefill_slots]

        # Set and arrange active batch ids for prefill
        seq_ids = [slot.id for slot in prefill_slots]
        seq_ids = torch.as_tensor(sorted(seq_ids), dtype=torch.int32)

        # Tokenize with padding
        padded_inputs = self.tokenizer(inputs,
                                       return_tensors="pt",
                                       padding=True)
        #  If needed truncate sequences to fit into the static dimensions
        seq_length = min(padded_inputs.input_ids.shape[-1], self.n_positions)
        input_ids = padded_inputs.input_ids[:, :seq_length]
        attention_mask = padded_inputs.attention_mask[:, :seq_length]
        n_active_seqs = len(input_ids)
        prefill_batch_size, prefill_context_len = input_ids.shape
        cache_ids = torch.arange(prefill_context_len).reshape(
            1, prefill_context_len).expand(
                n_active_seqs, prefill_context_len).mul(attention_mask)

        for i, slot in enumerate(prefill_slots):
            slot_input_ids = input_ids[i:i + 1, :]
            # Padded input ids are also required to set logits processors and stopping criterion
            selector = TokenSelector.create(slot_input_ids,
                                            slot.generation_config,
                                            self.model,
                                            self.n_positions,
                                            seed=slot.seed)
            slot_input_ids = slot_input_ids.squeeze().type(torch.int32)
            slot_attention_mask = attention_mask[i]
            slot_cache_ids = cache_ids[i]
            slot.reset(slot_input_ids, slot_attention_mask, selector,
                       slot_cache_ids)
        generation = self._generate_token(input_ids, attention_mask, cache_ids,
                                          seq_ids)
        # Reactivate previously active slots for the next decode.
        for slot in active_slots:
            slot.resume()
        logging.debug("Model ready for decoding")
        return generation

    def decode(self) -> List[Generation]:
        """Decode the specified requests.

        Return:
            A list of `Generation` for each request.
        """
        # Reconstruct input_ids and attention_mask from slots
        active_slots = self.get_slots_by_state(Slot.State.READY)
        input_ids = None
        attention_mask = None
        cache_ids = None
        seq_ids = None
        for i, slot in enumerate(active_slots):
            if input_ids is None:
                # Create blank inputs covering all active slots
                input_ids = torch.full([len(active_slots), 1],
                                       fill_value=self.tokenizer.eos_token_id,
                                       dtype=torch.int32)
            # input_ids are simply the tokens generated by the last decode or prefill requests (other tokens are cached)
            input_ids[i, 0] = slot.next_token
            if attention_mask is None:
                # Create default mask all active slots
                attention_mask = torch.zeros([len(active_slots), 1],
                                             dtype=torch.int32)
                attention_mask[:, -1] = 1
            if cache_ids is None:
                cache_ids = torch.zeros([len(active_slots), 1],
                                        dtype=torch.int32)
            cache_ids[i, :] = slot.cache_id
            if seq_ids is None:
                seq_ids = torch.zeros(len(active_slots), dtype=torch.int32)
            seq_ids[i] = slot.id
        if input_ids is None:
            raise ValueError(
                "Unable to decode tokens for non-prefilled batches (probably due to a previous failure)"
            )
        return self._generate_token(input_ids, attention_mask, cache_ids,
                                    seq_ids)

    def prepare_model_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
    ) -> Dict:
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": seq_ids
        }


class NaiveRollingBatchNeuronGenerator(NeuronGenerator):
    """A Generator for Neuron models recalculating KV cache each prefill."""

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, batch_size,
                 n_positions):
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        super().__init__(model, tokenizer, batch_size, n_positions)

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
            slot.assign(request, self.model.generation_config, self.tokenizer)
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
                slot.reset(slot_input_ids, slot_attention_mask, selector,
                           torch.LongTensor([0]))
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

    def prepare_model_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
    ) -> Dict:
        return self.model.prepare_inputs_for_generation(
            input_ids, attention_mask)
