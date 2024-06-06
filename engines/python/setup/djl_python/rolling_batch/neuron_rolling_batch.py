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
import logging
from typing import Optional, Any, List
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, FINISH_REASON_MAPPER
from djl_python.request_io import Token
from djl_python.transformers_neuronx_scheduler.optimum_neuron_scheduler import NaiveRollingBatchNeuronGenerator, ContinuousBatchingNeuronGenerator
from djl_python.properties_manager.tnx_properties import TnXGenerationStrategy, TransformerNeuronXProperties


class NeuronRollingBatch(RollingBatch):

    def __init__(self,
                 model,
                 tokenizer,
                 strategy: str = TnXGenerationStrategy.continuous_batching,
                 tnx_config: TransformerNeuronXProperties = None,
                 draft_model: Optional[Any] = None,
                 spec_length: Optional[int] = None) -> None:
        """
        Initializes the NeuronRollingBatch.

        :param model: the Neuron HuggingFace model
        :param batch_size: the maximum batch size required by model
        :param tokenizer: the tokenizer used by model
        :param n_positions: the maximum sequence size for model
        """
        self.strategy = strategy
        self._scheduler_class = ContinuousBatchingNeuronGenerator
        if draft_model or self.strategy == TnXGenerationStrategy.naive_rolling_batch:
            self._scheduler_class = NaiveRollingBatchNeuronGenerator

        super().__init__(tnx_config)
        self.scheduler = self._scheduler_class(model,
                                               tokenizer,
                                               tnx_config.batch_size,
                                               tnx_config.n_positions,
                                               draft_model=draft_model,
                                               spec_length=spec_length)

    def reset(self) -> None:
        """
        Aborts all requests.
        """
        self.scheduler.clear()
        super().reset()

    def get_tokenizer(self):
        return self.scheduler.tokenizer

    def parse_generation(self, generation, request, req_ids):
        if generation:
            is_last_token = False
            finish_reason = None
            if generation.generated_text is not None:
                is_last_token = True
                finish_reason = FINISH_REASON_MAPPER[int(
                    generation.generated_text.finish_reason.value)]
            if not is_last_token:
                req_ids.append(request.id)

            token_id = generation.token_id
            log_prob = generation.token_logprob
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()

            token = Token(
                token_id,
                "" if generation.token_is_special else generation.token_text,
                log_prob, generation.token_is_special)
            request.set_next_token(token,
                                   last_token=is_last_token,
                                   finish_reason=finish_reason)
        else:
            request.set_next_token("", last_token=False)
            req_ids.append(request.id)

    def append_speculated_generations(self, generation, request, req_ids):
        speculated_generation = generation.speculated_generations.dequeue()
        while speculated_generation is not None:
            self.parse_generation(speculated_generation, request, req_ids)
            speculated_generation = generation.speculated_generations.dequeue()

    @stop_on_any_exception
    def inference(self,
                  input_data: List[str],
                  parameters: List[dict],
                  adapters=None) -> list:
        """
        Loads new requests and gets output tokens from all currently active requests from
        the Neuron backend.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :param adapters: Optional adapters to apply

        :return: generated batch decoded tokens - list of dictionaries, one for
                 each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        if len(new_requests) > 0:
            generations = self.scheduler.prefill(new_requests)
        else:
            generations = self.scheduler.decode()
        generation_dict = {
            generation.request_id: generation
            for generation in generations
        }
        req_ids = []
        for request in self.active_requests:
            generation = generation_dict.get(request.id, None)
            self.parse_generation(generation, request, req_ids)
            if hasattr(generation, "speculated_generations"):
                self.append_speculated_generations(generation, request,
                                                   req_ids)

        # filter the requests that are stopped.
        self.scheduler.filter(req_ids)
        return self.postprocess_results()

    def preprocess_requests(self, requests: list):
        """
        Currently not applicable for Neuron.
        """
        raise NotImplementedError("Not implemented for Neuron rolling batcher")
