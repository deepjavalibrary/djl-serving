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

from abc import ABC, abstractmethod


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self, input_text: str, parameters: dict):
        """
        Initialize a request

        :param input_text: request's input text
        """
        self.input_text = input_text
        self.parameters = parameters
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


class RollingBatch(ABC):
    """
    This class initializes and maintains the SequenceBatchScheduler.
    Scheduler maintains the batch and also its search states such as past key values,
    attention masks and position ids for each decoding strategy requests.

    """

    def __init__(self, device):
        """
        Initializes the rolling batch scheduler.

        :param device: device to load the model
        """

        self.device = device
        self.pending_requests = []

    @abstractmethod
    def inference(self, input_data, parameters):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :return: generated batch decoded tokens
        """
        pass

    def get_new_requests(self, input_data, parameters, batch_size):
        new_requests = []
        pending_req_len = len(self.pending_requests)
        if batch_size > pending_req_len:
            for i in range(pending_req_len, batch_size):
                data = input_data[i]
                params = parameters[i] if i < len(parameters) else {}
                request = Request(data, params)
                self.pending_requests.append(request)
                new_requests.append(request)

        return new_requests

    @abstractmethod
    def preprocess_requests(self, requests):
        pass

    def postprocess_results(self, batch_size):
        results = []
        for i in range(batch_size):
            req = self.pending_requests[i]
            res = {"data": req.get_next_token(), "last": req.is_last_token()}
            results.append(res)

        for i in range(1, batch_size + 1):
            if self.pending_requests[batch_size - i].is_last_token():
                self.pending_requests.pop(batch_size - i)

        return results
