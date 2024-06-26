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
import os
from collections import OrderedDict, defaultdict

from lmi_dist.api import Request, RequestParams
from lmi_dist.arg_utils import VllmEngineArgs
from lmi_dist.init_engine import engine_from_args
from vllm.lora.request import LoRARequest
from vllm import SamplingParams

from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, filter_unused_generation_params
from djl_python.request_io import Token
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    get_speculative_decoding_metrics_record, update_request_cache_with_output,
    supports_speculative_decoding, get_lora_request_params, DTYPE_MAPPER,
    FINISH_REASON_MAPPER)
from djl_python.telemetry import telemetry_manager
from djl_python.properties_manager.lmi_dist_rb_properties import LmiDistRbProperties

_WARMUP_PREFILL_TOKENS = 4096
LMI_DIST_GENERATION_PARAMS = set(RequestParams().__dict__.keys()).union(
    set(SamplingParams().__dict__.keys()))


class LmiDistRollingBatch(RollingBatch):
    """
    LmiDistRollingBatch connects handler to LmiDist backend engine. It receives new
    requests from the handler and sends them to the backend when space is available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    def __init__(self, model_id_or_path: str, properties: dict, **kwargs):
        """
        Initializes the LmiDistRollingBatch.

        :param model_id_or_path (str): Currently unused since there is a copy inside properties
        :param properties (dict): other properties of the model, such as decoder strategy
        """
        self.lmi_dist_config = LmiDistRbProperties(**properties)
        self.model_type = getattr(kwargs.get("model_config", None),
                                  "model_type", None)
        super().__init__(self.lmi_dist_config)
        self.supports_speculative_decoding = supports_speculative_decoding()
        engine_kwargs = {}
        if self.supports_speculative_decoding:
            engine_kwargs[
                "draft_model"] = self.lmi_dist_config.speculative_draft_model
            engine_kwargs[
                "speculate_length"] = self.lmi_dist_config.speculative_length
            engine_kwargs[
                "draft_model_tp_size"] = self.lmi_dist_config.draft_model_tp_size
        engine_args = VllmEngineArgs(
            model=self.lmi_dist_config.model_id_or_path,
            tensor_parallel_size=self.lmi_dist_config.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[self.lmi_dist_config.dtype],
            seed=0,
            max_model_len=self.lmi_dist_config.max_model_len,
            enforce_eager=self.lmi_dist_config.enforce_eager,
            gpu_memory_utilization=self.lmi_dist_config.gpu_memory_utilization,
            max_num_batched_tokens=self.lmi_dist_config.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.lmi_dist_config.trust_remote_code,
            load_format=self.lmi_dist_config.load_format,
            quantization=self.lmi_dist_config.quantize,
            enable_lora=self.lmi_dist_config.enable_lora,
            max_loras=self.lmi_dist_config.max_loras,
            max_lora_rank=self.lmi_dist_config.max_lora_rank,
            lora_extra_vocab_size=self.lmi_dist_config.lora_extra_vocab_size,
            max_cpu_loras=self.lmi_dist_config.max_cpu_loras,
            revision=self.lmi_dist_config.revision,
            **engine_kwargs)

        kwargs = {}
        if self.lmi_dist_config.max_rolling_batch_prefill_tokens is None:
            kwargs["warmup_prefill_tokens"] = _WARMUP_PREFILL_TOKENS
        self.engine = engine_from_args(engine_args, **kwargs)
        self.request_cache = OrderedDict()
        self.lora_ids = defaultdict(lambda: len(self.lora_ids) + 1)

    def reset(self) -> None:
        """
        Aborts all requests
        """
        self.engine.reset(self.request_cache.keys())
        self.request_cache = OrderedDict()
        super().reset()

    def get_tokenizer(self):
        if "t5" == self.model_type:
            return self.engine.preprocessor.tokenizer
        return self.engine.preprocessor.tokenizer.tokenizer

    def translate_lmi_dist_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that lmi-dist recognizes.

        :param parameters (dict): Parameters pertaining to a specific request

        :return: The same parameters dict, but with lmi-dist style parameter names.
        """
        parameters["max_tokens"] = parameters.pop("max_new_tokens", 30)
        # If `do_sample` is not provided, force temperature=0.0, i.e. greedy
        # else set to user-provided value or default to 1.0
        if not parameters.pop('do_sample', False):
            parameters['temperature'] = 0.0
        else:
            parameters['temperature'] = parameters.get('temperature', 1.0)
        if "seed" in parameters.keys():
            parameters["seed"] = int(parameters["seed"])
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos_token")
        if "num_beams" in parameters.keys():
            parameters["best_of"] = parameters.pop("num_beams")
            parameters["use_beam_search"] = True
        if parameters.pop("decoder_input_details", False):
            parameters["prompt_logprobs"] = 1
        parameters["logprobs"] = parameters.get("logprobs", 1)
        parameters = filter_unused_generation_params(
            parameters,
            LMI_DIST_GENERATION_PARAMS,
            "lmi-dist",
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
            request_id = str(request.id)
            params = self.translate_lmi_dist_params(request.parameters)
            request_params = RequestParams(**params)
            lora_request_params = get_lora_request_params(
                request, self.lora_ids)
            lmi_dist_request = Request(
                id=request_id,
                prompt=request.input_text,
                params=request_params,
                lora_request=lora_request_params["lora_request"]
                if lora_request_params else None)
            self.engine.add_request(lmi_dist_request)
            self.request_cache[request_id] = {
                "curr_length": 0,
                "text": "",
                "output_token_texts": [],
                "cumulative_logprob": 0.0,
                "logprobs": [],
                "token_ids": [],
                "finished": False,
                "finish_reason": None,
                "num_generated_tokens": 0,
            }
        request_outputs = self.engine.step()

        # step 1: put result to cache
        for request_output in request_outputs:
            self.request_cache = update_request_cache_with_output(
                self.request_cache, request_output, self.get_tokenizer())
            # Record SD metrics
            completion_output = request_output.outputs[0]
            if (
                    self.lmi_dist_config.record_acceptance_rate
                    or self.lmi_dist_config.speculative_telemetry
            ) and self.lmi_dist_config.speculative_draft_model and request_output.finished:
                try:
                    if self.supports_speculative_decoding and completion_output.acceptance_history:
                        record = get_speculative_decoding_metrics_record(
                            completion_output, request_output)
                        if self.lmi_dist_config.record_acceptance_rate:
                            logging.info(f"Speculative Decoding {record}")
                        if self.lmi_dist_config.speculative_telemetry and os.environ.get(
                                "SAGEMAKER_SECURE_MODE") == "true":
                            telemetry_manager.record_speculative(record)
                except:
                    logging.debug('exception in sd telemetry, ignore...')

        # step 2: send result back
        finished_id = []
        for (key, cache), request in zip(self.request_cache.items(),
                                         self.active_requests):
            finish_reason = None
            prompt_tokens_details = None
            if cache["finished"]:
                finished_id.append(key)
                finish_reason = FINISH_REASON_MAPPER.get(
                    cache["finish_reason"], None)
                prompt_tokens_details = cache.get("prompt_tokens_details")
            text = cache["text"][cache["curr_length"]:]
            output_token_texts = [text] * len(cache['token_ids']) if not cache[
                'output_token_texts'] else cache['output_token_texts']
            if cache['token_ids']:
                # token id is not determined since there could be multiple token comes at the same time
                # only return the last one
                for token_id, token_text, logprob, in zip(
                        cache['token_ids'], output_token_texts,
                        cache['logprobs']):
                    token = Token(token_id, token_text, logprob)
                    request.set_next_token(token, cache["finished"],
                                           finish_reason,
                                           prompt_tokens_details)
            else:
                request.set_next_token("", cache["finished"], finish_reason,
                                       prompt_tokens_details)
            cache["curr_length"] = len(cache["text"])

        # step 3: clean finished requests
        for key in finished_id:
            self.request_cache.pop(key)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for lmi-dist.
        """
        raise NotImplementedError(
            "Not implemented for lmidist rolling batcher")
