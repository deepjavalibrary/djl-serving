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

from typing import List, Optional, Dict, Tuple

import torch
import logging
import threading
from abc import ABC, abstractmethod
from transformers_neuronx import base
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass

from djl_python.transformers_neuronx_scheduler.slot import Slot
from djl_python.rolling_batch.rolling_batch import filter_unused_generation_params
from djl_python.request import Request
from djl_python.transformers_neuronx_scheduler.token_selector import TokenSelector
from djl_python.transformers_neuronx_scheduler.speculation import (
    LMIDraftModelForSpeculation, LMIGreedyTokenAcceptor)
from djl_python.transformers_neuronx_scheduler.utils import (
    Generation, FinishReason, GeneratedText, TokenDecoder,
    SpeculatedGenerationsQueue)


@dataclass
class GenerationInputs:
    input_ids: torch.LongTensor
    attention_mask: Optional[torch.LongTensor] = None
    cache_ids: Optional[torch.LongTensor] = None
    seq_ids: Optional[torch.LongTensor] = None


@dataclass
class SlotRequestInputs:
    slot: Slot
    next_token: torch.LongTensor
    next_token_text: str
    next_log_prob: Optional[torch.LongTensor] = None


class NeuronGenerator(ABC):
    """A Generator for Neuron models."""

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        batch_size,
        n_positions,
        trim_cache=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.slots = [Slot(i) for i in range(batch_size)]
        self.batch_size = batch_size
        self.n_positions = n_positions
        self.trim_cache = trim_cache
        self.cache_ids = None
        self.prefill_generate = None
        self.decode_generate = None
        self.set_generation_methods()
        if not hasattr(self, "acceptor"):
            self.acceptor = None

    def get_slots_by_state(self, slot_state: Slot.State):
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        return slots[slot_state]

    def set_generation_methods(self):
        self.prefill_generate = self._generate_token
        self.decode_generate = self._generate_token

    @abstractmethod
    def _preprocess_prefill(self,
                            new_requests: List[Request]) -> GenerationInputs:
        """Prefill new requests.

        Return:
            Generation inputs for generate tokens.
        """

    def prefill(self, new_requests: List[Request]):
        """Prefill new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        active_slots = self.get_slots_by_state(Slot.State.READY)

        # pause the currently active slots, we will only run prefill on new slots
        for slot in active_slots:
            slot.pause()

        prefill_inputs = self._preprocess_prefill(new_requests)
        generation = self.prefill_generate(prefill_inputs)
        # Reactivate previously active slots for the next decode.
        for slot in active_slots:
            slot.resume()
        logging.debug("Model ready for decoding")
        return generation

    @abstractmethod
    def _preprocess_decode(self) -> GenerationInputs:
        """Decode the specified prefilled requests.

        Return:
            Generation inputs for batch generate tokens.
        """

    def decode(self) -> List[Generation]:
        """Decode the specified prefilled requests.

        Return:
            A list of `Generation` for each request.
        """
        inputs = self._preprocess_decode()
        return self.decode_generate(inputs)

    def prepare_model_inputs(self, inputs: GenerationInputs) -> Dict:
        """Prepare inputs for batching strategy
        Args:
            inputs (GenerationInputs): inputs tokenized tensor values

        Return:
            A dict of `Tensor`'s for each argument
        """
        return {
            "input_ids": inputs.input_ids,
            "cache_ids": inputs.cache_ids,
            "start_ids": inputs.seq_ids
        }

    def make_generations(
            self, slot_request: SlotRequestInputs
    ) -> Tuple[Generation, FinishReason]:
        """Build the generation, and finish reason for response
        """
        generated_text = None
        finish_reason = None
        if slot_request.slot.is_slot_eos_token(slot_request.next_token):
            finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
        elif slot_request.slot.stopped:
            finish_reason = FinishReason.FINISH_REASON_LENGTH
        if finish_reason is not None:
            # We must include the generated text for each finished sequence in the response
            generated_text = GeneratedText(
                text=slot_request.slot.generated_text,
                generated_tokens=slot_request.slot.generated_tokens,
                finish_reason=finish_reason,
                seed=slot_request.slot.seed)
        generation = Generation(
            request_id=slot_request.slot.request_id,
            prefill_tokens=None,
            token_id=slot_request.next_token,
            token_logprob=slot_request.next_log_prob,
            token_text=slot_request.next_token_text,
            token_is_special=(slot_request.next_token
                              in slot_request.slot.special_tokens),
            generated_text=generated_text,
            speculated_generations=SpeculatedGenerationsQueue())
        if finish_reason is not None:
            # mark the slot as available
            slot_request.slot.clear()
        return generation, finish_reason

    def _generate_token(self, inputs: GenerationInputs) -> List[Generation]:
        """Prepare inputs for batching strategy
        Args:
            inputs (GenerationInputs): inputs tokenized tensor values

        Return:
            A list of `Generation` for each request.
        """
        model_inputs = self.prepare_model_inputs(inputs)
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
            slot_input_ids = inputs.input_ids[i:i + 1, :]
            next_token, next_log_prob = slot.select(slot_input_ids,
                                                    next_token_logits)
            next_token_text = slot.decoder.decode(next_token.item())
            if self.trim_cache:
                slot.trim_cache_id()
            slot.append(next_token, next_token_text)
            slot_request = SlotRequestInputs(slot=slot,
                                             next_token=next_token,
                                             next_token_text=next_token_text,
                                             next_log_prob=next_log_prob)
            generation, _ = self.make_generations(slot_request)
            generations.append(generation)
        return generations

    @abstractmethod
    def _speculative_generate_token(
            self, inputs: GenerationInputs) -> List[Generation]:
        """Speculative forward method """

    @abstractmethod
    def _draft_target_generate(self,
                               inputs: GenerationInputs) -> List[Generation]:
        """Speculative forward method """

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
                 n_positions, **kwargs):
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        super().__init__(model,
                         tokenizer,
                         batch_size,
                         n_positions,
                         trim_cache=True)

    def _preprocess_prefill(self,
                            new_requests: List[Request]) -> GenerationInputs:
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
            slot.assign(request, self.model.generation_config, self.tokenizer,
                        self.acceptor)
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

        return GenerationInputs(input_ids=input_ids,
                                attention_mask=attention_mask,
                                cache_ids=cache_ids,
                                seq_ids=seq_ids)

    def _preprocess_decode(self):
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
        return GenerationInputs(input_ids=input_ids,
                                attention_mask=attention_mask,
                                cache_ids=cache_ids,
                                seq_ids=seq_ids)

    def _speculative_generate_token(
            self, inputs: GenerationInputs) -> List[Generation]:
        raise NotImplementedError(
            "Continuous batching does not support speculative decoding.")

    def _draft_target_generate(self,
                               inputs: GenerationInputs) -> List[Generation]:
        """Speculative forward method """
        raise NotImplementedError(
            "Continuous batching does not support speculative decoding.")


class NaiveRollingBatchNeuronGenerator(NeuronGenerator):
    """A Generator for Neuron models recalculating KV cache each prefill."""

    def __init__(self,
                 model,
                 tokenizer: PreTrainedTokenizerBase,
                 batch_size,
                 n_positions,
                 draft_model=None,
                 spec_length=4,
                 **kwargs):
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        if draft_model is None:
            super().__init__(model, tokenizer, batch_size, n_positions)
        else:
            if isinstance(draft_model, base.NeuronModelBase):
                draft_model = LMIDraftModelForSpeculation(draft_model)
            self.draft_model = draft_model
            self.spec_length = spec_length
            self.acceptor = LMIGreedyTokenAcceptor()
            self._threaded_generations = []
            super().__init__(model, tokenizer, batch_size, n_positions)
            self.set_speculative_generation_methods()

    def _preprocess_prefill(self,
                            new_requests: List[Request]) -> GenerationInputs:
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
        for request in new_requests:
            slot = empty_slots.pop()
            slot.assign(request, self.model.generation_config, self.tokenizer,
                        self.acceptor)
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
        return GenerationInputs(input_ids=input_ids,
                                attention_mask=attention_mask)

    def prefill(self, new_requests: List[Request]):
        """Prefill new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        active_slots = self.get_slots_by_state(Slot.State.READY)
        prefill_inputs = self._preprocess_prefill(new_requests)
        # Pause previously active slots during generation.
        # Their KV cache will be prefilled but new tokens will be ignored, as they
        # have already been generated and sent back in the last decode.
        for slot in active_slots:
            slot.pause()
        generation = self.prefill_generate(prefill_inputs)
        # Reactivate previously active slots for the next decode.
        for slot in active_slots:
            slot.resume()
        logging.debug("Model ready for decoding")
        return generation

    def _preprocess_decode(self):
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
        return GenerationInputs(input_ids=input_ids,
                                attention_mask=attention_mask)

    def set_speculative_generation_methods(self):
        self.prefill_generate = self._draft_target_generate
        self.decode_generate = self._speculative_generate_token

    def _batch_can_speculate(self, slots: List[Slot]) -> bool:
        for slot in slots:
            # Boundary condition: When number of leftover tokens to be generated is less than k,
            # we use the target model to use autoregressive generation the remaining tokens
            if slot.cache_id >= self.n_positions - 1 - self.spec_length:
                return False
        return True

    def _speculative_generate_token(
            self, inputs: GenerationInputs) -> List[Generation]:
        generations = []
        request_ids = []
        active_slots = self.get_slots_by_state(Slot.State.READY)
        if not self._batch_can_speculate(active_slots):
            return self._draft_target_generate(inputs)
        model_inputs = self.prepare_model_inputs(inputs)
        # Workaround until continuous batching and batch size > 1 is supported
        model_inputs['cache_ids'] = torch.LongTensor(
            [active_slots[0].tokens.shape[0] - 1])
        # Draft model sample
        draft_ids, draft_next_scores = self.draft_model(
            model_inputs['input_ids'], self.spec_length - 1, None,
            model_inputs['cache_ids'], model_inputs['start_ids'])
        # Create range of cache ids to be speculated
        cache_ids = torch.cat([
            # Add torch unsqueeze once we support batch size > 1
            torch.arange(model_inputs['cache_ids'][i],
                         model_inputs['cache_ids'][i] + draft_ids.shape[1] + 1)
            for i in range(model_inputs['cache_ids'].shape[0])
        ])
        input_ids = torch.cat([model_inputs['input_ids'], draft_ids], dim=1)
        target_next_scores = self.model.speculative_forward(
            input_ids=input_ids,
            cache_ids=cache_ids,
            start_ids=model_inputs['start_ids'],
            speculation_length=self.spec_length)
        for i, slot in enumerate(active_slots):
            request_id = slot.request_id
            request_ids.append(request_id)
            current = slot.tokens.shape[0] - 1
            # Fixed to match draft output
            slot_target_next_scores = target_next_scores.squeeze(dim=-1)

            # Accept speculative samples
            accepted_tokens, accepted_scores = slot.accept_speculated_tokens(
                draft_ids, draft_next_scores, slot_target_next_scores)

            if accepted_tokens.dim() != 2:
                accepted_tokens = accepted_tokens.view(1, -1)

            _, num_accepted = accepted_tokens.shape
            if self.n_positions - num_accepted < self.spec_length:
                accepted_tokens = accepted_tokens[:, :self.n_positions -
                                                  len(slot.tokens)]

            response_generation = None
            for index in range(num_accepted):
                next_token = accepted_tokens[:, index:index + 1]
                next_log_probs = accepted_scores[index:index + 1, :]
                next_token = torch.squeeze(next_token)
                next_log_probs = torch.squeeze(next_log_probs)
                next_token_text = slot.decoder.decode(next_token.item())
                slot.append(next_token, next_token_text)
                slot_request = SlotRequestInputs(
                    slot=slot,
                    next_token=next_token,
                    next_token_text=next_token_text,
                    next_log_prob=next_log_probs)
                generation, finish_reason = self.make_generations(slot_request)
                if response_generation is not None:
                    response_generation.speculated_generations.enqueue(
                        generation)
                else:
                    response_generation = generation
                    generations.append(response_generation)
                current = current + 1
                if current >= self.n_positions - 1 or finish_reason:
                    break

            # If we accepted all tokens then we need to insert the last draft
            # token into the KV cache since it was generated but never executed
            # TODO: Determine how to extend this once batch > 1 is supported
            if num_accepted == self.spec_length:
                self.draft_model(accepted_tokens[:, -2:-1], 1,
                                 torch.tensor([current - 1]),
                                 model_inputs['start_ids'])

        return generations

    def _draft_target_generate(
        self,
        inputs: GenerationInputs,
    ) -> List[Generation]:
        model_inputs = self.prepare_model_inputs(inputs)

        def generate_draft(_model_inputs):
            _, _ = self.draft_model(_model_inputs["input_ids"], 1, None,
                                    _model_inputs["cache_ids"],
                                    _model_inputs["start_ids"])

        def generate_target(_generations, _model_inputs):
            _generations.append(self.model(
                **_model_inputs,
                return_dict=True,
            ))

        target_task = threading.Thread(target=generate_target,
                                       args=(self._threaded_generations,
                                             model_inputs))
        draft_task = threading.Thread(target=generate_draft,
                                      args=(model_inputs, ))
        target_task.start()
        draft_task.start()
        # Threaded execution of the draft models first forward and the target models first forward
        # improves time to first token
        draft_task.join()
        target_task.join()
        outputs = self._threaded_generations.pop()
        self._threaded_generations.clear()  # Should be empty already

        generations = []
        request_ids = []
        active_slots = self.get_slots_by_state(Slot.State.READY)
        for i, slot in enumerate(active_slots):
            request_id = slot.request_id
            request_ids.append(request_id)
            next_token_logits = outputs.logits[i:i + 1, -1, :]
            slot_input_ids = inputs.input_ids[i:i + 1, :]
            next_token, next_log_prob = slot.select(slot_input_ids,
                                                    next_token_logits)
            next_token_text = slot.decoder.decode(next_token.item())
            slot.append(next_token, next_token_text)
            slot_request = SlotRequestInputs(slot=slot,
                                             next_token=next_token,
                                             next_token_text=next_token_text,
                                             next_log_prob=next_log_prob)
            generation, finish_reason = self.make_generations(slot_request)
            generations.append(generation)
        return generations
