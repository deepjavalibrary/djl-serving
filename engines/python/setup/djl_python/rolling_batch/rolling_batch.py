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
from abc import ABC, abstractmethod
from typing import List

from djl_python.properties_manager.properties import Properties
from djl_python.request import Request
from djl_python.request_io import Token
from djl_python.utils import IdCounter

FINISH_REASON_MAPPER = ["length", "eos_token", "stop_sequence"]


def filter_unused_generation_params(parameters: dict,
                                    allowed_params: set,
                                    backend: str,
                                    remove_unused_params: bool = False):
    unused_params = set(parameters.keys()) - allowed_params
    if len(unused_params) > 0:
        logging.warning(
            f"The following parameters are not supported by {backend} with rolling batch: {unused_params}. The "
            f"supported parameters are {allowed_params}")
        if remove_unused_params:
            for param in unused_params:
                parameters.pop(param)
    return parameters


def stop_on_any_exception(func):
    """
    Decorator that handles errors sent from backend
    """

    def try_catch_handling(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.exception("Rolling batch inference error")
            for request in self.active_requests:
                token = Token(-1,
                              "",
                              log_prob=-1,
                              special_token=True,
                              error_msg=str(e))
                request.set_next_token(token,
                                       last_token=True,
                                       finish_reason="error")
            response = self.postprocess_results()
            self.reset()
            return response

    return try_catch_handling


class RollingBatch(ABC):
    """
    This class initializes and maintains the SequenceBatchScheduler.
    Scheduler maintains the batch and also its search states such as past key values,
    attention masks and position ids for each decoding strategy requests.

    """

    def __init__(self, configs: Properties):
        """
        Initializes the rolling batch scheduler.

        :param  passed while loading the model
        """

        self.active_requests: List[Request] = []
        self.req_id_counter = IdCounter()
        self.configs = configs

    def reset(self):
        self.active_requests = []
        self.req_id_counter.reset()

    def get_tokenizer(self):
        """
        :return: the tokenizer used for inference
        """
        raise RuntimeError("get_tokenizer function not supported")

    @abstractmethod
    def inference(self, requests: List[Request]) -> List:
        """
        Performs prefill and decode operations for the batch.

        :param requests: List[Request] List of requests

        :return: generated batch decoded tokens
        """
        pass

    def get_new_requests(self, requests: List[Request]) -> List[Request]:
        """
        Adds requests to the batch when there is availability

        :param requests: List[Request] List of requests

        :return: list of current active requests (including those that have just been added)
        """
        total_req_len = len(self.active_requests)
        batch_size = len(requests)
        if batch_size > total_req_len:
            for i in range(total_req_len, batch_size):
                self.active_requests.append(requests[i])
        return self.active_requests[total_req_len:]

    @abstractmethod
    def preprocess_requests(self, requests: List[Request]):
        """
        Converts requests into specific formats that are required by specific backends.

        :param requests (List[Request]): requests that will be sent to the backend after preprocessing
        """
        pass

    def postprocess_results(self) -> List[dict]:
        """
        Returns most recent produced token by each request in a list of dicts

        :return: a list of dicts, each one containing token and metadata
        """
        results = []
        for i in range(len(self.active_requests)):
            req = self.active_requests[i]
            res = {
                "data": req.get_next_token(),
                "last": req.is_last_token(),
                "content_type": req.get_content_type()
            }
            req.reset_next_token()
            results.append(res)

        self.active_requests = [
            req for req in self.active_requests if not req.is_last_token()
        ]

        if len(self.active_requests) == 0:
            self.req_id_counter.reset()

        return results
