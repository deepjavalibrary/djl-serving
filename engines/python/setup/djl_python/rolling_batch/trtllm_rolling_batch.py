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
import tensorrt_llm_toolkit

from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception
from djl_python.request_io import Token


class TRTLLMRollingBatch(RollingBatch):
    """
    TRTLLMRollingBatch connects the TensorRT-LLM handler to the TensorRT-LLM backend. It receives new
    requests from the handler and sends them to the backend when there is space available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    def __init__(self, model_id_or_path: str, properties: dict,
                 configs: TensorRtLlmProperties) -> None:
        """
        Initializes the TRTLLMRollingBatch

        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(configs)
        if not configs.mpi_mode:
            raise AssertionError(
                f"Need mpi_mode to start tensorrt llm RollingBatcher")
        self.model = tensorrt_llm_toolkit.init_inference(
            model_id_or_path, **properties)
        self.request_cache = {}

    def get_tokenizer(self):
        return self.model.tokenizer

    def reset(self) -> None:
        """
        Stops all current requests and resets state of rolling batch portion of handler
        """
        # TODO: delete running requests of Triton will cause backend crashes
        # bare with memory leaks of failed requests
        self.request_cache.clear()
        super().reset()

    def translate_triton_params(self, parameters: dict) -> dict:
        """
        Helper function to convert DJL Serving parameter names to Triton
        parameter names that TensorRT-LLM recognizes.

        :param parameters: Parameters pertaining to a specific request

        :return: The same parameters dict, but with TensorRT-LLM style parameter names.
        """
        if "request_output_len" not in parameters.keys():
            parameters["request_output_len"] = parameters.pop(
                "max_new_tokens", 30)
        if "top_k" in parameters.keys():
            parameters["runtime_top_k"] = parameters.pop("top_k")
        if "top_p" in parameters.keys():
            parameters["runtime_top_p"] = parameters.pop("top_p")
        if "seed" in parameters.keys():
            parameters["random_seed"] = int(parameters.pop("seed"))
        if parameters.pop("do_sample", False):
            parameters["runtime_top_k"] = parameters.get("runtime_top_k", 5)
            parameters["runtime_top_p"] = parameters.get("runtime_top_p", 0.85)
            parameters["temperature"] = parameters.get("temperature", 0.8)
        if "length_penalty" in parameters.keys():
            parameters['len_penalty'] = parameters.pop('length_penalty')
        parameters["streaming"] = parameters.pop(
            "stream", parameters.get("streaming", True))
        stop = parameters.pop("stop", None)
        if stop:
            # stop_sequences is translated to stop_words_list in tensorrt_llm_toolkit
            parameters["stop_sequences"] = stop
        return parameters

    @stop_on_any_exception
    def inference(self,
                  input_data: list[str],
                  parameters: list[dict],
                  adapters=None) -> list:
        """
        Loads new requests into the batch when there is availability, and gets output tokens from the backend
        asynchronously.

        :param input_data: List of input prompts.
        :param parameters: List of settings pertaining to each request.
        :param adapters: List of adapters inputs for each request in a batch

        :return results: List of dictionaries, one for each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        # add pending requests to active requests list
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new active requests
        for request in new_requests:
            param = self.translate_triton_params(request.parameters)
            output_len = param["request_output_len"]
            response = self.model.generate(request.input_text, **param)
            self.request_cache[request.id] = {
                "response": response,
                "out_length": output_len,
                "cumulative_logprob": 0
            }

        # step 1: loop the active requests to send result
        for request in self.active_requests:
            trt_resp = self.request_cache[request.id]["response"]
            generation = trt_resp.fetch()
            log_prob = generation.cum_logprob - self.request_cache[
                request.id]["cumulative_logprob"]
            self.request_cache[
                request.id]["cumulative_logprob"] = generation.cum_logprob
            token = Token(generation.token_id, generation.token_text, log_prob,
                          None)
            if generation.finished:
                finish_reason = "eos_token" if generation.seq_length < self.request_cache[
                    request.id]["out_length"] else "length"
                request.set_next_token(token, generation.finished,
                                       finish_reason)
                self.request_cache.pop(request.id)
            else:
                request.set_next_token(token, generation.finished)

        return self.postprocess_results()

    def preprocess_requests(self, requests: list):
        """
        Currently not applicable for TensorRT-LLM.
        """
        raise NotImplementedError(
            "Not implemented for tensorrtllm rolling batcher")
