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
import tensorrt_llm_toolkit
from djl_python.rolling_batch.rolling_batch import RollingBatch


class TRTLLMRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the TRTLLMRollingBatch.
        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(-1, **kwargs)
        self.model = tensorrt_llm_toolkit.init_inference(
            model_id_or_path, **kwargs)
        self.request_cache = {}

    def reset(self):
        """
        Stops all current requests and resets state of rolling batch portion of handler
        """
        for req in self.request_cache.values():
            self.model.delete_request(req)
        self.request_cache.clear()
        super().reset()

    def translate_triton_params(self, parameters):
        parameters["request_output_len"] = int(parameters.get("max_new_tokens", 128))
        if "top_k" in parameters.keys():
            parameters["runtime_top_k"] = parameters.pop("top_k")
        if "top_p" in parameters.keys():
            parameters["runtime_top_p"] = parameters.pop("top_p")
        if "seed" in parameters.keys():
            parameters["random_seed"] = int(parameters.pop("seed"))
        if "do_sample" in parameters.keys():
            parameters.pop("do_sample")
            logging.info(
                "do_sample is not used for trtllm,"
                " please use sample params (e.g top_p, temperature) to allow sampling"
            )
        parameters["streaming"] = parameters.get("streaming", True)
        return parameters

    def inference(self, input_data, parameters):
        batch_size = len(input_data)
        # add pending requests to active requests list
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new active requests
        for request in new_requests:
            param = self.translate_triton_params(request.parameters)
            response = self.model.generate(request.input_text, **param)
            self.request_cache[request.id] = response

        # step 1: loop the active requests to send result
        for request in self.active_requests:
            trt_req = self.request_cache[request.id]
            output_text, complete = trt_req.fetch()
            request.set_next_token(output_text, self.output_formatter,
                                   complete)
            if complete:
                self.request_cache.pop(request.id)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        raise NotImplementedError(
            "Not implemented for tensorrtllm rolling batcher")
