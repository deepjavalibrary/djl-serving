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
from collections import OrderedDict

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import random_uuid
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}


class VLLMRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the VLLMRollingBatch.
        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(-1, **kwargs)
        self.dtype = properties.get("dtype", 'auto')
        max_batched_prefill_tokens = None
        if properties.get("engine") != "Python":
            raise AssertionError(
                f"Need python engine to start vLLM RollingBatcher")
        if "max_rolling_batch_prefill_tokens" in properties:
            max_batched_prefill_tokens = int(
                properties.get("max_rolling_batch_prefill_tokens"))
        tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", None))
        args = EngineArgs(model=model_id_or_path,
                          tensor_parallel_size=tensor_parallel_degree,
                          dtype=DTYPE_MAPPER[self.dtype],
                          seed=0,
                          max_num_batched_tokens=max_batched_prefill_tokens,
                          trust_remote_code=kwargs.get("trust_remote_code",
                                                       False),
                          quantization=properties.get("quantize", None))
        self.engine = LLMEngine.from_engine_args(args)
        self.request_cache = OrderedDict()

    def reset(self):
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = OrderedDict()
        super().reset()

    @stop_on_any_exception
    def inference(self, input_data, parameters):
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = random_uuid()
            request.parameters.pop('seed', None)
            request.parameters.pop('do_sample', None)
            if "max_new_tokens" in request.parameters.keys():
                request.parameters["max_tokens"] = request.parameters.pop(
                    "max_new_tokens")
            sampling_params = SamplingParams(**request.parameters)
            self.engine.add_request(request_id, request.input_text,
                                    sampling_params)
            self.request_cache[request_id] = {
                "curr_length": 0,
                "text": "",
                "finished": False
            }
        request_outputs = self.engine.step()
        # step 1: put result to cache
        for request_output in request_outputs:
            req_id = request_output.request_id
            self.request_cache[req_id]["text"] = request_output.outputs[0].text
            if len(request_output.outputs) > 1:
                logging.warning(
                    f"Finding more than 1 output for single request {len(request_output.outputs)}"
                    f"Beam search is not supported yet, use first output by default"
                )
            self.request_cache[req_id]["finished"] = request_output.finished
        # step 2: send result back
        finished_id = []
        for (key, cache), request in zip(self.request_cache.items(),
                                         self.active_requests):
            request.set_next_token(cache["text"][cache["curr_length"]:],
                                   self.output_formatter, cache["finished"])
            cache["curr_length"] = len(cache["text"])
            if cache["finished"]:
                finished_id.append(key)
        # step 3: clean finished requests
        for key in finished_id:
            self.request_cache.pop(key)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
