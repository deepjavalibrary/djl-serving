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

from vllm import LLMEngine
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.utils import random_uuid, AtomicCounter

from djl_python.request import Request
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, filter_unused_generation_params
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    update_request_cache_with_output, create_lora_request, get_lora_request,
    get_engine_args_from_config, get_prompt_inputs)
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from typing import List, Optional

# FIXME: Once all vllm versions are past 0.6.0 we can move to just struct_fields
VLLM_GENERATION_PARAMS = set(SamplingParams().__struct_fields__).union(
    {"use_beam_search"})


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
        parameters["output_kind"] = RequestOutputKind.DELTA
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
            request_params = dict()
            if request.adapter is not None:
                adapter_name = request.adapter.get_property("name")
                request_params["lora_request"] = get_lora_request(
                    adapter_name, self.lora_requests)
            self.request_cache[request_id] = {
                "request_output": request.request_output
            }
            if "use_beam_search" in params:
                beam_search_params = self.to_beam_search_params(params)
                request_output = self.beam_search(request_id=request_id,
                                                  prompt=prompt_inputs,
                                                  params=beam_search_params)
                self.request_cache = update_request_cache_with_output(
                    self.request_cache, request_output, self.get_tokenizer())
            else:
                sampling_params = SamplingParams(**params)
                self.engine.add_request(request_id=request_id,
                                        inputs=prompt_inputs,
                                        params=sampling_params,
                                        **request_params)
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

    def to_beam_search_params(self, parameters: dict) -> BeamSearchParams:
        return BeamSearchParams(beam_width=parameters.get("n", 1),
                                max_tokens=parameters["max_tokens"],
                                ignore_eos=parameters.get("ignore_eos", False),
                                temperature=parameters.get("temperature", 0.0),
                                length_penalty=parameters.get(
                                    "length_penalty", 1.0))

    def beam_search(
        self,
        request_id: str,
        prompt: PromptType,
        params: BeamSearchParams,
    ):
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty

        prompt_token_ids = prompt if "prompt_token_ids" in prompt else self.get_tokenizer(
        ).encode(prompt["prompt"])
        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            self.get_tokenizer().eos_token_id, length_penalty)

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids, cum_logprob=0)
        ]
        completed = []

        for _ in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            beam_search_request_id = f"beam_search-{random_uuid()}"

            # output = self.generate(prompts_batch, beam_search_params,
            #                        request_id)
            for i, individual_prompt in enumerate(prompts_batch):
                self.engine.add_request(
                    request_id=f"{beam_search_request_id}-{i}",
                    prompt=individual_prompt,
                    params=beam_search_params)
            output = self.engine.step()
            output = self.engine.step()

            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        new_beam = BeamSearchSequence(
                            tokens=current_beam.tokens + [token_id],
                            cum_logprob=current_beam.cum_logprob +
                            logprob_obj.logprob)

                        if token_id == self.get_tokenizer().eos_token_id and \
                                not ignore_eos:
                            completed.append(new_beam)
                        else:
                            new_beams.append(new_beam)

            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            beam.text = self.get_tokenizer().decode(
                beam.tokens[tokenized_length:])

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt,
            outputs=[
                CompletionOutput(
                    text=beam.text,
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens,
                    index=i,
                    logprobs=beam.cum_logprob,
                ) for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        return beam_search_output

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
