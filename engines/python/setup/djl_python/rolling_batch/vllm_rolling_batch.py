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
import json
import logging
from collections import OrderedDict

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import random_uuid
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token

from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}

FINISH_REASON_MAPPER = {
    "length": "length",
    "stop": "eos_token",
    "abort": "abort"
}


class VLLMRollingBatch(RollingBatch):
    """
    VLLMRollingBatch connects the handler to the backend (VLLM inference). It receives new
    requests from the handler and sends them to the backend when there is space available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    # TODO: Make properties is the only parameter, after refactoring all rolling batch handlers
    def __init__(self, model_id_or_path: str, properties: dict, **kwargs):
        """
        Initializes the VLLMRollingBatch.

        :param model_id_or_path (str): Currently unused since there is a copy inside properties
        :param properties (dict): other properties of the model, such as decoder strategy
        """
        self.vllm_configs = VllmRbProperties(**properties)
        super().__init__(waiting_steps=self.vllm_configs.waiting_steps,
                         output_formatter=self.vllm_configs.output_formatter)
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
            draft_model=self.vllm_configs.speculative_draft_model,
            speculate_length=self.vllm_configs.speculative_length,
            draft_model_tp_size=self.vllm_configs.draft_model_tp_size,
            revision=self.vllm_configs.revision)
        self.engine = LLMEngine.from_engine_args(args)
        self.request_cache = OrderedDict()

    def reset(self):
        """
        Aborts all requests
        """
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = OrderedDict()
        super().reset()

    def translate_vllm_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that VLLM recognizes.

        :param parameters (dict): Parameters pertaining to a specific request

        :return: The same parameters dict, but with VLLM style parameter names.
        """
        parameters.pop('seed', None)
        parameters.pop('do_sample', None)
        if "max_new_tokens" in parameters.keys():
            parameters["max_tokens"] = parameters.pop("max_new_tokens")
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos")
        return parameters

    @stop_on_any_exception
    def inference(self, input_data: list[str], parameters: list[dict]) -> list:
        """
        Adds new requests and gets output tokens from the backend.

        :param input_data (list[str]): List of input prompts.
        :param parameters (list[dict]): List of settings pertaining to each request.

        :return results (list): List of dictionaries, one for each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = random_uuid()
            params = self.translate_vllm_params(request.parameters)
            sampling_params = SamplingParams(**params)
            self.engine.add_request(request_id, request.input_text,
                                    sampling_params)
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
            req_id = request_output.request_id
            self.request_cache[req_id]["id"] = request_output.outputs[
                0].token_ids[-1]
            self.request_cache[req_id]["text"] = request_output.outputs[0].text
            # calculate log_prob of the token based on the diff between two cumulative log probs
            self.request_cache[req_id]["log_prob"] = request_output.outputs[
                0].cumulative_logprob - self.request_cache[req_id][
                    "cumulative_logprob"]
            self.request_cache[req_id][
                "cumulative_logprob"] = request_output.outputs[
                    0].cumulative_logprob
            self.request_cache[req_id][
                "finish_reason"] = request_output.outputs[0].finish_reason
            if len(request_output.outputs) > 1:
                logging.warning(
                    f"Finding more than 1 output for single request {len(request_output.outputs)}"
                    f"Beam search is not supported yet, use first output by default"
                )
            self.request_cache[req_id]["finished"] = request_output.finished
            # Record SD metrics
            completion_output = request_output.outputs[0]
            if self.vllm_configs.record_acceptance_rate and request_output.finished and completion_output.acceptance_history:
                record = {}
                record["id"] = req_id
                if len(completion_output.acceptance_history) > 0:
                    record["mean_acceptance"] = 1.0 * sum(
                        completion_output.acceptance_history) / len(
                            completion_output.acceptance_history)
                else:
                    record["mean_acceptance"] = 0
                record["prompt_size"] = len(request_output.prompt_token_ids)
                record["output_size"] = len(completion_output.token_ids)
                logging.info(f"Speculative Decoding {record}")
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
                request.set_next_token(token, self.output_formatter,
                                       cache["finished"], finish_reason)
            else:
                request.set_next_token("", self.output_formatter,
                                       cache["finished"], finish_reason)
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
