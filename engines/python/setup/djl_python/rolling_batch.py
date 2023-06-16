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
import torch

DEFAULT_SEARCH_ALGORITHM = 'greedy'

MODEL_TYPE_2_BLOCK = {'bloom': BloomBlock}


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self, input_text: str):
        """
        Initialize a request

        :param input_text: request's input text
        """
        self.input_text = input_text
        self.next_token = None
        self.last_token = False

    def set_next_token(self, next_token: str, last_token: bool = False):
        """
        Sets the newly generated token.

        :param next_token: next token to be set.
        :param last_token: whether this token is the last of the sequence.
        """
        self.next_token = next_token
        self.last_token = last_token

    def get_next_token(self) -> str:
        """
        Gets the token generated for the request.

        :return: next_token
        """
        return self.next_token

    def is_last_token(self) -> bool:
        """
        Whether the generated token is the last one

        :return: whether last token of the sequence.
        """
        return self.last_token


class RollingBatch:
    """
    This class initializes and maintains the SequenceBatchScheduler.
    Scheduler maintains the batch and also its search states such as past key values,
    attention masks and position ids for each decoding strategy requests.

    """

    def __init__(self, model, tokenizer, config, device, properties):
        """
        Initializes the rolling batch scheduler.

        :param model: loaded model
        :param tokenizer: tokenizer of the model
        :param config: configuration of the model
        :param device: model loaded device
        :param properties: other properties of the model, such as decoder strategy
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        self.pending_requests = []

    def inference(self, input_data, parameters):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :return: generated batch decoded tokens
        """
        batch_size = len(input_data)
        new_requests = self._get_new_requests(input_data, parameters,
                                              batch_size)
        self._merge_request(new_requests)
        results = []
        for i in range(len(input_data)):
            req = self.pending_requests[i]
            res = {"data": req.get_next_token(), "last": req.is_last_token()}
            results.append(res)

        for i in range(1, batch_size + 1):
            if self.pending_requests[batch_size - i].is_last_token():
                self.pending_requests.pop(batch_size - i)

        return results

    def _get_new_requests(self, input_data, parameters, batch_size):
        Requests = namedtuple('Requests',
                              ['input_texts', 'search_configs', 'request_ids'])
        new_requests = Requests(defaultdict(list), defaultdict(list),
                                defaultdict(list))

        pending_req_len = len(self.pending_requests)
        if batch_size > pending_req_len:
            req_id_counter = _calculate_req_id_counter(self.scheduler)
            for i in range(pending_req_len, batch_size):
                data = input_data[i]
                self.pending_requests.append(Request(data))

                search_algorithm = parameters[i].get('decoding_strategy',
                                                     self.search_algorithm)
                new_requests.input_texts[search_algorithm].append(data)

                search_config = self._construct_search_config(parameters[i])
                new_requests.search_configs[search_algorithm].append(
                    search_config)
                new_requests.request_ids[search_algorithm].append(
                    req_id_counter)
                req_id_counter += 1

        return new_requests

    def _merge_request(self, new_requests):

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
