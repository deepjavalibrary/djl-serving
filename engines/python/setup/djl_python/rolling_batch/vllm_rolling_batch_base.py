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
from collections import OrderedDict
import logging

from abc import abstractmethod
from vllm import SamplingParams

from djl_python.rolling_batch.rolling_batch import RollingBatch, Request, Token, stop_on_any_exception
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties

FINISH_REASON_MAPPER = {
    "length": "length",
    "stop": "eos_token",
    "abort": "abort"
}

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}


class VllmRollingBatchBase(RollingBatch):
    """
    This class implements shared inference functionality that will be inherited by
    vllm and lmidist-v2
    """

    def __init__(self, engine_config: VllmRbProperties):
        self.engine_config = engine_config
        self.request_cache = OrderedDict()
        super().__init__(waiting_steps=self.engine_config.waiting_steps,
                         output_formatter=self.engine_config.output_formatter)

    @abstractmethod
    def init_engine(self):
        """
        Initializes backend engine - vLLM/lmidist_v2
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Aborts all requests
        """
        pass

    @stop_on_any_exception
    def inference(self, input_data: list[str], parameters: list[dict]) -> list:
        """
        Adds new requests and gets output tokens from the backend.

        :param input_data (list[str]): List of input prompts.
        :param parameters (list[str]): List of settings pertaining to each request.

        :return results (list): List of dictionaries, one for each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)

        # step 0: register new requests to engine
        for request in new_requests:
            request_id = self.get_request_id(request)
            params = self.translate_to_engine_params(request.parameters)
            sampling_params = SamplingParams(**params)
            self.add_request(request_id, request.input_text, sampling_params)
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
            self._record_speculative_decoding_metrics(request_output, req_id)

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

    @abstractmethod
    def add_request(self, request_id: str, prompt: str,
                    sampling_params: SamplingParams):
        """
        Adds request to backend engine
        """
        pass

    @abstractmethod
    def translate_to_engine_params(self, parameters: dict) -> dict:
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that backend engine recognizes.

        :param parameters: Parameters pertaining to a specific request

        :return: The same parameters dict, but with VLLM style parameter names.
        """
        pass

    @abstractmethod
    def get_request_id(self, request: Request):
        """
        Get request id that will be set to backend engine request
        """
        pass

    def _record_speculative_decoding_metrics(self, request_output, req_id):
        completion_output = request_output.outputs[0]
        if self.engine_config.record_acceptance_rate and request_output.finished and completion_output.acceptance_history:
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
