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
from collections import OrderedDict, defaultdict

from vllm import LLMEngine, SamplingParams
from vllm.utils import random_uuid, AtomicCounter

from djl_python.request import Request
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, filter_unused_generation_params
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    update_request_cache_with_output, create_lora_request, get_lora_request,
    get_engine_args_from_config, get_prompt_inputs)
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from typing import List, Optional

# FIXME: Once all vllm versions are past 0.6.0 we can move to just struct_fields
VLLM_GENERATION_PARAMS = set(SamplingParams().__struct_fields__) if hasattr(
    SamplingParams(), "__struct_fields__") else set(
        SamplingParams().__dict__.keys())


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
        super().__init__(self.vllm_configs)
        args = get_engine_args_from_config(self.vllm_configs)
        self.engine = LLMEngine.from_engine_args(args)
        self.request_cache = OrderedDict()
        self.lora_id_counter = AtomicCounter(0)
        self.lora_requests = {}
        self.is_mistral_tokenizer = self.vllm_configs.tokenizer_mode == 'mistral'

    def get_tokenizer(self):
        return self.engine.tokenizer.tokenizer

    def get_huggingface_model_config(self):
        return self.engine.model_config.hf_config

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

        # If `do_sample` is not provided, force temperature=0.0, i.e. greedy
        # else set to user-provided value or default to 1.0
        if not parameters.pop('do_sample', False):
            parameters['temperature'] = 0.0
        else:
            parameters['temperature'] = parameters.get('temperature', 1.0)
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos_token")
        if "num_beams" in parameters.keys():
            parameters["best_of"] = parameters.pop("num_beams")
            parameters["use_beam_search"] = True
        if parameters.pop("decoder_input_details", False):
            parameters["prompt_logprobs"] = 1

        # if n is not explicitly set when best_of is set, we return `best_of` values sequences for tgi compatibility.
        if "best_of" in parameters.keys():
            if "n" not in "best_of":
                parameters["n"] = parameters["best_of"]

        if "top_n_tokens" in parameters.keys():
            parameters["logprobs"] = parameters.pop("top_n_tokens")
        else:
            parameters["logprobs"] = parameters.get("logprobs", 1)
        parameters = filter_unused_generation_params(parameters,
                                                     VLLM_GENERATION_PARAMS,
                                                     "vllm",
                                                     remove_unused_params=True)
        return parameters

    @stop_on_any_exception
    def inference(self, new_requests: List[Request]) -> List:
        """
        Adds new requests and gets output tokens from the backend.

        :param new_requests: List[Request] List of requests

        :return results: List of dictionaries, one for each request, that contain output tokens and other data.
        """
        self.add_new_requests(new_requests)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = random_uuid()
            prompt_inputs = get_prompt_inputs(request)
            params = self.translate_vllm_params(request.parameters)
            sampling_params = SamplingParams(**params)
            request_params = dict()
            if request.adapter is not None:
                adapter_name = request.adapter.get_property("name")
                request_params["lora_request"] = get_lora_request(
                    adapter_name, self.lora_requests)
            self.engine.add_request(request_id=request_id,
                                    inputs=prompt_inputs,
                                    params=sampling_params,
                                    **request_params)
            self.request_cache[request_id] = {
                "request_output": request.request_output
            }
        request_outputs = self.engine.step()

        # step 1: put result to cache and request_output
        for request_output in request_outputs:
            self.request_cache = update_request_cache_with_output(
                self.request_cache, request_output, self.get_tokenizer())

        for request in self.active_requests:
            request_output = request.request_output
            if request_output.finished:
                request.last_token = True

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for VLLM.
        """
        raise NotImplementedError("Not implemented for vLLM rolling batcher")

    def add_lora(self,
                 lora_name: str,
                 lora_path: str,
                 long_lora_max_len: Optional[int] = None):
        """
        Add LoRA adapter.
        """
        lora_id = self.lora_id_counter.inc(1)
        lora_request = create_lora_request(lora_name,
                                           lora_id,
                                           lora_path,
                                           long_lora_max_len=long_lora_max_len)
        self.lora_requests[lora_request.lora_name] = lora_request
        return self.engine.add_lora(lora_request)

    def remove_lora(self, lora_name):
        """
        Remove LoRA adapter.
        """
        lora_request = get_lora_request(lora_name, self.lora_requests)
        return self.engine.remove_lora(lora_request.lora_int_id)

    def pin_lora(self, lora_name):
        """
        Pin LoRA adapter.
        """
        lora_request = get_lora_request(lora_name, self.lora_requests)

        # To pin an adapter, adapter has to be registered already (by calling add_lora()).
        # If trying to pin an adapter that is not registered, we will get "LoRA is not registered" error.
        # However, registered adapters are maintained by LRUCache
        # and may be evicted if the number of adapters exceed capacity (max_cpu_loras).
        # So there will be two scenarios:
        # 1) An adapter is evicted, call add_lora() is necessary to avoid error.
        # 2) An adapter is not evicted, call add_lora() is not necessary.
        # But since whether an adapter is evicted is not exposed outside of engine,
        # and add_lora() in this case will take negligible time, we will still call add_lora().
        loaded = self.engine.add_lora(lora_request)
        return loaded and self.engine.pin_lora(lora_request.lora_int_id)
