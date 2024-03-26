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
import logging
from collections import OrderedDict, defaultdict

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import random_uuid
from vllm.lora.request import LoRARequest
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token, filter_unused_generation_params
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    update_request_cache_with_output, get_lora_request_params, DTYPE_MAPPER,
    FINISH_REASON_MAPPER)
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties

VLLM_GENERATION_PARAMS = set(SamplingParams().__dict__.keys())


class VLLMRollingBatch(RollingBatch):
    """
    VLLMRollingBatch connects the handler to the backend (VLLM inference). It receives new
    requests from the handler and sends them to the backend when there is space available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    # TODO: Make properties is the only parameter, after refactoring all rolling batch handlers
    def __init__(self, model_id_or_path: str, properties: dict,
                 **kwargs) -> None:
        """
        Initializes the VLLMRollingBatch.

        :param model_id_or_path: Currently unused since there is a copy inside properties
        :param properties: other properties of the model, such as decoder strategy
        """
        self.vllm_configs = VllmRbProperties(**properties)
        super().__init__(waiting_steps=self.vllm_configs.waiting_steps)
        args = EngineArgs(
            model=self.vllm_configs.model_id_or_path,
            tensor_parallel_size=self.vllm_configs.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[self.vllm_configs.dtype],
            seed=0,
            max_model_len=self.vllm_configs.max_model_len,
            enforce_eager=self.vllm_configs.enforce_eager,
            gpu_memory_utilization=self.vllm_configs.gpu_memory_utilization,
            max_num_batched_tokens=self.vllm_configs.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.vllm_configs.trust_remote_code,
            load_format=self.vllm_configs.load_format,
            quantization=self.vllm_configs.quantize,
            enable_lora=self.vllm_configs.enable_lora,
            max_loras=self.vllm_configs.max_loras,
            max_lora_rank=self.vllm_configs.max_lora_rank,
            lora_extra_vocab_size=self.vllm_configs.lora_extra_vocab_size,
            max_cpu_loras=self.vllm_configs.max_cpu_loras,
            revision=self.vllm_configs.revision)
        self.engine = LLMEngine.from_engine_args(args)
        self.request_cache = OrderedDict()
        self.lora_ids = defaultdict(lambda: len(self.lora_ids) + 1)

    def get_tokenizer(self):
        return self.engine.tokenizer.tokenizer

    def reset(self) -> None:
        """
        Aborts all requests
        """
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = OrderedDict()
        super().reset()

    def translate_vllm_params(self, parameters: dict) -> dict:
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that VLLM recognizes.

        :param parameters: Parameters pertaining to a specific request

        :return: The same parameters dict, but with VLLM style parameter names.
        """
        parameters["max_tokens"] = parameters.pop("max_new_tokens", 30)
        if "seed" in parameters.keys():
            parameters["seed"] = int(parameters["seed"])
        if not parameters.pop('do_sample', False):
            # if temperature is zero, vLLM does greedy sampling
            parameters['temperature'] = 0
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos_token")
        if "num_beams" in parameters.keys():
            parameters["best_of"] = parameters.pop("num_beams")
            parameters["use_beam_search"] = True
        parameters = filter_unused_generation_params(parameters,
                                                     VLLM_GENERATION_PARAMS,
                                                     "vllm",
                                                     remove_unused_params=True)
        return parameters

    @stop_on_any_exception
    def inference(self,
                  input_data: list[str],
                  parameters: list[dict],
                  adapters=None) -> list:
        """
        Adds new requests and gets output tokens from the backend.

        :param input_data: List of input prompts.
        :param parameters: List of settings pertaining to each request.
        :param adapters: List of adapters inputs for each request in a batch

        :return results: List of dictionaries, one for each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data,
                                             parameters,
                                             batch_size,
                                             adapters=adapters)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = random_uuid()
            params = self.translate_vllm_params(request.parameters)
            sampling_params = SamplingParams(**params)
            request_params = get_lora_request_params(request, self.lora_ids)
            self.engine.add_request(request_id, request.input_text,
                                    sampling_params, **request_params)
            self.request_cache[request_id] = {
                "curr_length": 0,
                "text": "",
                "cumulative_logprob": 0.0,
                "log_prob": 0.0,
                "finished": False,
                "finish_reason": None
            }
        request_outputs = self.engine.step()

        # step 1: put result to cache
        for request_output in request_outputs:
            self.request_cache = update_request_cache_with_output(
                self.request_cache, request_output)

        # step 2: send result back
        finished_id = []
        for (key, cache), request in zip(self.request_cache.items(),
                                         self.active_requests):
            finish_reason = None
            if cache["finished"]:
                finished_id.append(key)
                finish_reason = FINISH_REASON_MAPPER.get(
                    cache["finish_reason"], None)
            text = cache["text"][cache["curr_length"]:]
            if len(text) > 0:
                # token id is not determined since there could be multiple token comes at the same time
                # only return the last one
                token = Token(cache['id'], text, cache["log_prob"])
                request.set_next_token(token, cache["finished"], finish_reason)
            else:
                request.set_next_token("", cache["finished"], finish_reason)
            cache["curr_length"] = len(cache["text"])

        # step 3: clean finished requests
        for key in finished_id:
            self.request_cache.pop(key)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for VLLM.
        """
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
