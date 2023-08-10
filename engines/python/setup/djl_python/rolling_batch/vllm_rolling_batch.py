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
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import random_uuid
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception


class VLLMRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the VLLMRollingBatch.
        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(-1, **kwargs)
        self.dtype = kwargs.pop("dtype", 'auto')
        if properties.get("engine") != "Python":
            raise AssertionError(
                f"Need python engine to start vLLM RollingBatcher")
        tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", None))
        if kwargs:
            logging.warning(f"kwargs content are dropped {kwargs}")
        args = EngineArgs(
            model=model_id_or_path,
            tensor_parallel_size=tensor_parallel_degree,
            dtype=self.dtype,
            seed=0,
        )
        self.engine = LLMEngine.from_engine_args(args)
        self.request_cache = {}

    def reset(self):
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = {}
        super().reset()

    @stop_on_any_exception
    def inference(self, input_data, parameters):
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        for request in new_requests:
            request_id = random_uuid()
            request.parameters.pop('seed', None)
            sampling_params = SamplingParams(**request.parameters)
            self.engine.add_request(request_id, request.input_text,
                                    sampling_params)
            self.request_cache[request_id] = {"curr_length": 0}
        request_outputs = self.engine.step()
        for request_output, request in zip(request_outputs,
                                           self.pending_requests):
            req_id = request_output.request_id
            gen_text = request_output.outputs[0].text
            if len(request_output.outputs) > 0:
                logging.warning(
                    f"Finding more than 1 output for single request {len(request_output.outputs)}"
                    f"Beam search is not supported yet, use first output by default"
                )
            request.set_next_token(
                gen_text[self.request_cache[req_id]["curr_length"]:],
                self.output_formatter, request_output.finished)
            self.request_cache[req_id]["curr_length"] = len(gen_text)
            if request_output.finished:
                self.request_cache.pop(req_id)

        return self.postprocess_results(batch_size)

    def preprocess_requests(self, requests):
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
