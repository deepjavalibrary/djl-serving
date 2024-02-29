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
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token, FINISH_REASON_MAPPER
from lmi_dist.models import get_model
from lmi_dist.utils.parameters import (
    NextTokenChooserParameters,
    StoppingCriteriaParameters,
)
import lmi_dist
from lmi_dist.utils.types import (Batch, Request, Generation)

import torch

from djl_python.properties_manager.lmi_dist_rb_properties import LmiDistRbProperties

QUANTIZATION_SUPPORT_ALGO = ["bitsandbytes8", "bitsandbytes", "gptq", "awq"]


class LmiDistRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, properties, **kwargs):
        """
        Initializes the LmiDistRollingBatch.

        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        :param kwargs passed while loading the model
        """

        self.lmi_dist_configs = LmiDistRbProperties(**properties)

        super().__init__(
            waiting_steps=self.lmi_dist_configs.waiting_steps,
            output_formatter=self.lmi_dist_configs.output_formatter)
        self.batch_cls = None
        self._init_model(self.lmi_dist_configs.model_id_or_path,
                         self.lmi_dist_configs.draft_model_id)
        self.batch_id_counter = 0
        self.cache = {}

    def reset(self):
        self.cache.clear()
        self.batch_id_counter = 0
        super().reset()

    def _init_model(self, model_id_or_path, draft_model_id=None):
        sharded = self.lmi_dist_configs.tensor_parallel_degree > 1
        quantize = self.lmi_dist_configs.quantize
        if quantize is not None:
            quantize = quantize.value
        self.model = get_model(
            model_id_or_path,
            revision=self.lmi_dist_configs.revision,
            sharded=sharded,
            quantize=quantize,
            dtype=self.lmi_dist_configs.dtype,
            trust_remote_code=self.lmi_dist_configs.trust_remote_code)
        self.draft_model = get_model(
            draft_model_id,
            revision=self.lmi_dist_configs.revision,
            sharded=False,
            quantize=quantize,
            dtype=self.lmi_dist_configs.dtype,
            trust_remote_code=self.lmi_dist_configs.trust_remote_code,
            is_draft_model=True) if draft_model_id else None
        self.batch_cls = self.model.batch_type
        self._warmup()

    def _warmup(self):
        max_batch_prefill_tokens = self.lmi_dist_configs.max_rolling_batch_prefill_tokens

        input_length = 512
        n_tokens = 0
        req_id = 0
        requests = []

        while n_tokens < max_batch_prefill_tokens:
            truncate = min(input_length, max_batch_prefill_tokens - n_tokens)
            requests.append(
                lmi_dist.utils.types.Request(
                    id=req_id,
                    inputs='_test ' * input_length,
                    parameters=NextTokenChooserParameters(
                        temperature=0.9,
                        repetition_penalty=1.2,
                        top_k=10,
                        top_p=0.9,
                        typical_p=0.9,
                        do_sample=False,
                        seed=0),
                    stopping_parameters=StoppingCriteriaParameters(
                        stop_sequences=[], max_new_tokens=2),
                    truncate=truncate,
                    prefill_logprobs=True))
            n_tokens += input_length
            req_id += 1

        batch = self.batch_cls.get_batch(
            Batch(id=0, requests=requests, size=len(requests)),
            self.model.config,
            self.model.tokenizer,
            self.lmi_dist_configs.torch_dtype,
            self.lmi_dist_configs.device,
            spec_length=self.lmi_dist_configs.spec_length
            if self.draft_model else 0)
        max_batch_total_tokens = self.model.warmup(batch, self.draft_model)
        if max_batch_total_tokens is not None and self.lmi_dist_configs.device == 0:
            logging.info(
                f"The max total sequence length is {max_batch_total_tokens}")

    def release_cache(self):
        self.model.release_cache()

    @stop_on_any_exception
    def inference(self, input_data, parameters, adapters=None):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :param adapters: List of adapters inputs for each request in a batch
        :return: generated batch decoded tokens
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        new_batch = self.preprocess_requests(new_requests)
        if new_batch or self.cache:
            self._prefill_and_decode(new_batch)
        return self.postprocess_results()

    def _prefill_and_decode(self, new_batch):
        """
        About the text quality issue in Nov. 2023, it was temporarily solved by [RP#1189: Fix lmi_dist garbage output
        issue](https://github.com/deepjavalibrary/djl-serving/pull/1189). The root cause of this issue is now
        believed to be found. It should be the buggy memory management; the batch.release() was called inside
        __del__ (see the code is in flash_causal_lm.py in lmi-dist repo), which won't be triggered until the reference
        count of the batch is 0. So paged cached memory allocated to the batch won't be freed in time. Also,
        the self.block_tables = None is missing, which will cause repetitive freeing of the paged cache memory,
        and the non-uniqueness in batch.block_tables_tensor.

        In [RP#1189: Fix lmi_dist garbage output issue](https://github.com/deepjavalibrary/djl-serving/pull/1189),
        even though the algorithm remains equivalent, how the reference count to batch is different, so when __del__
        is called is different. That's why this PR temporarily fixes the text quality issue. Now with this controlled
        batch.release(), reverting this PR, the text quality issue is believed to disappear.
        """

        # prefill step
        if new_batch:
            batch = new_batch
            generations, next_batch = self.model.generate_token(
                batch,
                draft_model=self.draft_model if self.draft_model else None)
            if next_batch is not None:
                self.cache[next_batch.batch_id] = next_batch
        else:
            # Get batches out of cache
            batches = [x for x in self.cache.values()]
            self.cache.clear()
            if len(batches) > 1:
                batch = self.model.batch_type.concatenate(batches)
            else:
                batch = batches[0]
            generations, next_batch = self.model.generate_token(
                batch,
                draft_model=self.draft_model if self.draft_model else None)
            if next_batch is not None:
                self.cache[next_batch.batch_id] = next_batch

        req_ids = []
        for request in self.active_requests:
            generation = generations.get(request.id, None)
            if generation:
                is_last_token = False
                finish_reason = None
                if generation.generated_text is not None:
                    is_last_token = True
                    finish_reason = FINISH_REASON_MAPPER[int(
                        generation.generated_text.finish_reason.value)]
                if not is_last_token:
                    req_ids.append(request.id)

                token_id = generation.token_id
                log_prob = generation.token_logprob
                if isinstance(token_id, torch.Tensor):
                    token_id = token_id.item()
                if isinstance(log_prob, torch.Tensor):
                    log_prob = log_prob.item()
                token = Token(token_id, generation.token_text_no_special,
                              log_prob, generation.token_is_special)
                request.set_next_token(token,
                                       self.output_formatter,
                                       last_token=is_last_token,
                                       finish_reason=finish_reason)
            else:
                request.set_next_token("",
                                       self.output_formatter,
                                       last_token=False)

        # filter the requests that are stopped.
        if self.cache and batch.batch_id in self.cache:
            self.cache[batch.batch_id] = self.cache[batch.batch_id].filter(
                req_ids)

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
                max_new_tokens=param.get("max_new_tokens", 30),
                ignore_eos_token=param.get("ignore_eos_token", False))

            request = lmi_dist.utils.types.Request(
                id=r.id,
                inputs=r.input_text,
                parameters=parameters,
                stopping_parameters=stop_parameters)
            truncate = param.get("truncate", None)
            if truncate is not None:
                request.truncate = truncate
            preprocessed_requests.append(request)

        if preprocessed_requests:
            batch = Batch(id=self.batch_id_counter,
                          requests=preprocessed_requests,
                          size=len(preprocessed_requests))
            self.batch_id_counter += 1

            return self.batch_cls.get_batch(
                batch,
                self.model.config,
                self.model.tokenizer,
                self.lmi_dist_configs.torch_dtype,
                self.lmi_dist_configs.device,
                # spec_dec parameters
                self.lmi_dist_configs.spec_length if self.draft_model else 0)
        else:
            return None
