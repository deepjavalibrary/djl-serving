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

from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token, FINISH_REASON_MAPPER, filter_unused_generation_params
from deepspeed.external.lmi_dist.utils.parameters import (
    NextTokenChooserParameters,
    StoppingCriteriaParameters,
)
from deepspeed.external.lmi_dist.utils.types import (Batch, Request)
from deepspeed.inference.engine import InferenceEngine
from deepspeed.inference.rolling_batch import DeepSpeedRollingBatchGeneration

DEEPSPEED_GENERATION_PARAMETERS = set(
    NextTokenChooserParameters().__dict__.keys()).union(
        set(StoppingCriteriaParameters().__dict__.keys()))


class DeepSpeedRollingBatch(RollingBatch):

    def __init__(self, model: InferenceEngine, properties, **kwargs):
        """
        Initializes the DeepSpeedRollingBatch.

        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        :param kwargs passed while loading the model
        """

        super().__init__(**kwargs)
        self.error_requests = []
        self.properties = properties
        self.batch_cls = None
        self.batch_id_counter = 0
        self.rolling_batch = DeepSpeedRollingBatchGeneration(
            model=model,
            tokenizer=kwargs.get("tokenizer"),
            max_batch_size=kwargs.get("max_batch_size"),
            max_seq_len=kwargs.get("max_seq_len"))

    def reset(self):
        self.error_requests = []
        self.batch_id_counter = 0
        super().reset()

    def _warmup(self, **kwargs):
        pass

    def postprocess_results(self):
        results = []
        err_reqs = dict((r.id, err) for r, err in self.error_requests)
        for req in self.active_requests:
            if req.id in err_reqs.keys():
                res = {
                    "data":
                    "",
                    "last":
                    True,
                    "code":
                    424,
                    "error":
                    f"Request: `{req.input_text}` failed due to: {err_reqs.get(req.id)}"
                }
            else:
                res = {
                    "data": req.get_next_token(),
                    "last": req.is_last_token()
                }
            results.append(res)

        # add empty tokens to pending requests
        for i in range(len(self.active_requests),
                       len(self.active_requests) + len(self.pending_requests)):
            res = {"data": "", "last": False}
            results.append(res)

        self.active_requests = [
            req for req in self.active_requests
            if not req.is_last_token() and req.id not in err_reqs.keys()
        ]
        self.pending_requests = [
            req for req in self.pending_requests
            if req.id not in err_reqs.keys()
        ]

        if len(self.active_requests) + len(self.pending_requests) == 0:
            self.req_id_counter = 0

        return results

    @stop_on_any_exception
    def inference(self, input_data, parameters, adapters=None):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :param adapters: List of adapters inputs for each request in a batch
        :return: generated batch decoded tokens
        """
        new_requests = self.get_new_requests(input_data, parameters,
                                             len(input_data))
        new_batch = self.preprocess_requests(new_requests)
        if new_batch or len(self.active_requests) > 0:
            self._prefill_and_decode(new_batch)
        return self.postprocess_results()

    def _prefill_and_decode(self, new_batch):
        if new_batch:
            batch = new_batch
            generations, error_requests = self.rolling_batch.prefill_batch(
                batch)
            self.error_requests = error_requests
        else:
            generations = self.rolling_batch.generate_token()
        for request in self.active_requests:
            generation = None
            # TODO(mohaan): Change generations to a Dict with request id index
            filtered_gens = list(
                filter(lambda g: g.request_id == request.id, generations))
            if len(filtered_gens) > 0:
                generation = filtered_gens[0]
            if generation:
                is_last_token = False
                finish_reason = None
                if generation.generated_text is not None:
                    is_last_token = True
                    finish_reason = FINISH_REASON_MAPPER[int(
                        generation.generated_text.finish_reason.value)]
                token = Token(
                    generation.token_id, ""
                    if generation.token_is_special else generation.token_text,
                    generation.token_logprob.item(),
                    generation.token_is_special)
                request.set_next_token(token,
                                       last_token=is_last_token,
                                       finish_reason=finish_reason)
            else:
                request.set_next_token("", last_token=False)

    def preprocess_requests(self, requests, **kwargs):
        preprocessed_requests = []
        for r in requests:
            param = r.parameters
            parameters = NextTokenChooserParameters(
                temperature=param.get("temperature", 1.0),
                repetition_penalty=param.get("repetition_penalty", 1.0),
                top_k=param.get("top_k", 0),
                top_p=param.get("top_p", 1.0),
                typical_p=param.get("typical_p", 1.0),
                do_sample=param.get("do_sample", False),
                seed=int(param.get("seed", 0)))
            stop_parameters = StoppingCriteriaParameters(
                stop_sequences=param.get("stop_sequences", []),
                max_new_tokens=param.get("max_new_tokens", 30))
            truncate = param.get("truncate", None)

            filter_unused_generation_params(param,
                                            DEEPSPEED_GENERATION_PARAMETERS,
                                            "deepspeed")
            request = Request(id=r.id,
                              inputs=r.input_text,
                              parameters=parameters,
                              stopping_parameters=stop_parameters)
            if truncate is not None:
                request.truncate = truncate
            preprocessed_requests.append(request)

        if preprocessed_requests:
            batch = Batch(id=self.batch_id_counter,
                          requests=preprocessed_requests,
                          size=len(preprocessed_requests))
            self.batch_id_counter += 1

            return batch
        else:
            return None
