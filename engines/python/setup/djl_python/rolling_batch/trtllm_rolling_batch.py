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
from transformers import AutoTokenizer
from collections import OrderedDict
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception


class TRTLLMRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the TRTLLMRollingBatch.
        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(-1, **kwargs)
        self.model = tensorrt_llm_toolkit.init_inference(
            model_id_or_path)  # not sure kwargs okay or not
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left",
                                                       revision=kwargs.get(
                                                           'revision', None))
        self.request_cache = OrderedDict()

    def reset(self):
        """
        Stops all current requests and resets state of rolling batch portion of handler
        """
        for key in self.request_cache.keys():
            continue
            # todo stop the asynchronous inference
        self.request_cache = OrderedDict()
        super().reset()

    def inference(self, input_data, parameters):
        batch_size = len(input_data)
        # add pending requests to active requests list
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # register new active requests
        for request in new_requests:
            result = self.model.generate(request.input_text,
                                         streaming=[True
                                                    ])  #request parameters?
            self.request_cache[request.id] = {
                "response_obj": result,
                "curr_length": 0,
                "text": "",
                "finished": False
            }

        # obtain new tokens in all active requests
        finished_ids = set()
        for request in self.active_requests:
            cached_request = self.request_cache[request.id]
            output_text, complete = cached_request["response_obj"].fetch()
            cached_request["curr_length"] += 1
            cached_request["text"] += output_text
            request.set_next_token(output_text, self.output_formatter,
                                   complete)
            if (complete):
                finished_ids.add(request.id)
                cached_request["finished"] = True

        # remove finished requests
        for finished_id in finished_ids:
            del self.request_cache[finished_id]

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        raise NotImplementedError(
            "Not implemented for tensorrtllm rolling batcher")
