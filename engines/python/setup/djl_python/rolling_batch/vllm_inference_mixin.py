#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from vllm import SamplingParams
from vllm.utils import random_uuid


from djl_python.rolling_batch.rolling_batch import stop_on_any_exception, Token

FINISH_REASON_MAPPER = {
    "length": "length",
    "stop": "eos_token",
    "abort": "abort"
}


class VllmInferenceMixin():

    @stop_on_any_exception
    def inference_impl(self, input_data, parameters):
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = random_uuid()
            params = self.translate_vllm_params(request.parameters)
            sampling_params = SamplingParams(**params)
            self._add_request_to_engine(request_id, request.input_text, sampling_params)
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