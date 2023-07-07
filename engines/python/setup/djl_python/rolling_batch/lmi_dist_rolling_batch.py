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

from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception
from transformers import AutoConfig
from lmi_dist.models import get_model
from lmi_dist.utils.parameters import (
    NextTokenChooserParameters,
    StoppingCriteriaParameters,
)
import lmi_dist
from lmi_dist.utils.types import (Batch, Request, Generation)

import torch

QUANTIZATION_SUPPORT_ALGO = ["bitsandbytes"]


class LmiDistRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the LmiDistRollingBatch.

        :param model_id_or_path: model id or path
        :param device: model loaded device
        :param properties: other properties of the model, such as decoder strategy
        :param kwargs passed while loading the model
        """

        super().__init__(device, **kwargs)
        self.properties = properties
        self.batch_cls = None
        self._init_model(kwargs, model_id_or_path)
        self.batch_id_counter = 0
        self.cache: Batch = None

    def _init_model(self, kwargs, model_id_or_path):
        self.config = AutoConfig.from_pretrained(model_id_or_path, **kwargs)
        sharded = int(self.properties.get("tensor_parallel_degree", "-1")) > 1
        quantize = self.properties.get("quantize", None)
        dtype = self.properties.get("dtype", None)
        if quantize is not None and dtype is not None:
            raise ValueError(
                f"Can't set both dtype: {dtype} and quantize: {quantize}")
        if quantize is not None and quantize not in QUANTIZATION_SUPPORT_ALGO:
            raise ValueError(
                f"Invalid value for quantize: {quantize}. Valid values are: {QUANTIZATION_SUPPORT_ALGO}"
            )
        if quantize is None and dtype == "int8":
            quantize = "bitsandbytes"
        self.model = get_model(
            model_id_or_path,
            revision=None,
            sharded=sharded,
            quantize=quantize,
            trust_remote_code=kwargs.get("trust_remote_code"))
        self.batch_cls = self.model.batch_type

    @stop_on_any_exception
    def inference(self, input_data, parameters):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :return: generated batch decoded tokens
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        new_batch = self.preprocess_requests(new_requests)
        self._prefill_and_decode(new_batch)
        return self.postprocess_results(batch_size)

    def _prefill_and_decode(self, new_batch):
        # prefill step
        if new_batch:
            generations, prefill_next_batch = self.model.generate_token(
                new_batch)

            if self.cache:
                decode_generations, decode_next_batch = self.model.generate_token(
                    self.cache)
                generations.extend(decode_generations)

                # concatenate with the existing batch of the model
                if decode_next_batch:
                    self.cache = self.model.batch_type.concatenate(
                        [prefill_next_batch, decode_next_batch])
                else:
                    self.cache = prefill_next_batch
            else:
                self.cache = prefill_next_batch
        else:
            generations, next_batch = self.model.generate_token(self.cache)
            self.cache = next_batch

        generation_dict = {
            generation.request_id: generation
            for generation in generations
        }

        req_ids = []
        for request in self.pending_requests:
            generation = generation_dict[request.id]
            is_last_token = generation.generated_text is not None
            if not is_last_token:
                req_ids.append(request.id)

            if generation.token_is_special:
                request.set_next_token("", last_token=is_last_token)
            elif self.output_formatter is not None:
                request.set_next_token(self.output_formatter(
                    [generation.token_text]),
                                       last_token=is_last_token)
            else:
                request.set_next_token(generation.token_text,
                                       last_token=is_last_token)

        # filter the requests that are stopped.
        if self.cache:
            self.cache = self.cache.filter(req_ids)

    def preprocess_requests(self, requests, **kwargs):
        preprocessed_requests = []
        for r in requests:
            param = r.parameters
            parameters = NextTokenChooserParameters(
                temperature=param.get(
                    "temperature",
                    0.5),  # TODO: Find a better place to put default values
                repetition_penalty=param.get("repetition_penalty", 1.0),
                top_k=param.get("top_k", 4),
                top_p=param.get("top_p", 1.0),
                typical_p=param.get("typical_p", 1.0),
                do_sample=param.get("do_sample", False),
                seed=int(param.get("seed", 0)))
            stop_parameters = StoppingCriteriaParameters(
                stop_sequences=param.get("stop_sequences", []),
                max_new_tokens=param.get("max_new_tokens", 30))

            preprocessed_requests.append(
                lmi_dist.utils.types.Request(
                    id=r.id,
                    inputs=r.input_text,
                    parameters=parameters,
                    stopping_parameters=stop_parameters))

        if preprocessed_requests:
            batch = Batch(id=self.batch_id_counter,
                          requests=preprocessed_requests,
                          size=len(preprocessed_requests))
            self.batch_id_counter += 1

            return self.batch_cls.get_batch(
                batch, self.model.tokenizer,
                kwargs.get("torch_dtype", torch.float16), self.device)
        else:
            return None
