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

from djl_python.scheduler import HuggingfaceBlock, BloomBlock, SearchConfig, SeqBatchScheduler
from collections import namedtuple, defaultdict
from djl_python.rolling_batch.rolling_batch import RollingBatch, Request
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

import torch

MODEL_TYPE_2_BLOCK = {'bloom': BloomBlock}
DEFAULT_SEARCH_ALGORITHM = 'greedy'


class SchedulerRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the rolling batch scheduler.

        :param model_id_or_path: model id or path
        :param device: model loaded device
        :param properties: other properties of the model, such as decoder strategy
        :param kwargs passed while loading the model
        """

        super().__init__(device)
        self._init_model_and_tokenizer(kwargs, model_id_or_path)
        self._init_scheduler(properties)

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

        preprocessed_new_requests = self.preprocess_requests(new_requests)
        self._prefill_and_decode(preprocessed_new_requests)
        return self.postprocess_results(batch_size)

    def preprocess_requests(self, requests):
        Requests = namedtuple('Requests',
                              ['input_texts', 'search_configs', 'request_ids'])
        new_requests = Requests(defaultdict(list), defaultdict(list),
                                defaultdict(list))

        req_id_counter = _calculate_req_id_counter(self.scheduler)
        for request in requests:
            parameters = request.parameters
            search_algorithm = parameters.get('decoding_strategy',
                                              self.search_algorithm)
            new_requests.input_texts[search_algorithm].append(
                request.input_text)

            search_config = self._construct_search_config(parameters)
            new_requests.search_configs[search_algorithm].append(search_config)
            new_requests.request_ids[search_algorithm].append(req_id_counter)
            req_id_counter += 1

        return new_requests

    def _init_model_and_tokenizer(self, kwargs, model_id_or_path):
        self.config = AutoConfig.from_pretrained(model_id_or_path,
                                                 kwargs=kwargs)
        architectures = self.config.architectures
        if architectures and architectures[0].endswith(
                "ForConditionalGeneration"):
            raise ValueError('Seq2Seq model is not supported by scheduler')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, **kwargs)

        if self.device:
            self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_scheduler(self, properties):
        lm_block_cls = MODEL_TYPE_2_BLOCK.get(self.config.model_type,
                                              HuggingfaceBlock)
        self.lm_block = lm_block_cls(self.model)
        self.search_config = SearchConfig(
            eos_token_id=self.tokenizer.eos_token,
            pad_token_id=self.tokenizer.pad_token)
        self.search_algorithm = properties.get('decoding_strategy',
                                               DEFAULT_SEARCH_ALGORITHM)
        self.scheduler = SeqBatchScheduler(self.lm_block,
                                           self.search_algorithm,
                                           self.search_config)

    def _prefill_and_decode(self, new_requests):

        for search_algorithm in new_requests.request_ids.keys():
            request_ids = new_requests.request_ids[search_algorithm]
            if request_ids:
                input_texts = new_requests.input_texts[search_algorithm]
                search_configs = new_requests.search_configs[search_algorithm]

                # Prefills search states for each request and merges to the existing batch
                self.scheduler.add_request(
                    input_ids=self._get_input_ids(input_texts=input_texts),
                    request_uids=_get_request_ids_tensor(
                        request_ids=request_ids),
                    search_algorithm=search_algorithm,
                    search_configs=search_configs)

        # Decoding step. Generates a token for all the requests in a batch.
        generated_token_ids, request_ids, exit_req_ids = self.scheduler.inference_call(
        )

        # TODO: Deleting the finished results here temporarily
        for request_id in exit_req_ids:
            if request_id in self.scheduler.results:
                del self.scheduler.results[request_id]

        generated_tokens = self.tokenizer.batch_decode(generated_token_ids)

        for request_id, generated_token, request in zip(
                request_ids, generated_tokens, self.pending_requests):
            is_last_token = (request_id in exit_req_ids)
            request.set_next_token(generated_token, last_token=is_last_token)

    def _get_input_ids(self, input_texts):
        input_ids = self.tokenizer(input_texts,
                                   return_tensors="pt",
                                   padding=True).input_ids
        if self.device is not None:
            input_ids = input_ids.to(self.device)
        return input_ids

    def _construct_search_config(self, parameters):
        return SearchConfig(
            max_new_tokens=parameters.get('max_new_tokens',
                                          self.search_config.max_new_seqlen),
            top_k=parameters.get('top_k', self.search_config.topk),
            penalty_alpha=parameters.get('penalty_alpha',
                                         self.search_config.alpha),
            num_beams=parameters.get('num_beams', self.search_config.beam),
            do_sample=parameters.get('do_sample', self.search_config.sampling),
            top_p=parameters.get('top_p', self.search_config.topk),
            temperature=parameters.get('temperature',
                                       self.search_config.temperature))


def _get_request_ids_tensor(request_ids):
    request_ids_tensor = torch.tensor(request_ids)
    request_ids_tensor = torch.reshape(request_ids_tensor,
                                       (len(request_ids), 1))
    return request_ids_tensor


def _calculate_req_id_counter(scheduler):
    if scheduler:
        request_ids = scheduler.get_request_ids()
        if request_ids:
            return request_ids[-1] + 1
    return 0
