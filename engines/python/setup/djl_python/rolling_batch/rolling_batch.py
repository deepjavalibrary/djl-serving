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
from typing import List, Optional

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
            logging.exception(
                f"Rolling batch inference error. There are {len(self.active_requests)} requests impacted. Dumping the impacted requestIds"
            )
            for request in self.active_requests:
                logging.info(
                    f"[RequestId={request.get_client_request_id()}] impacted by rolling batch error"
                )
                error_message = "exception occurred during rolling batch inference"
                token = Token(-1,
                              "",
                              log_prob=-1,
                              special_token=True,
                              error_msg=error_message)
                request.set_next_token(token,
                                       last_token=True,
                                       finish_reason="error")
                request.set_error_message(error_message)
                # TODO: make configurable
                request.set_error_code(424)
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

    def get_model_config(self):
        """
        :return: the model config if available
        """
        raise RuntimeError("get_model_config must be implemented by subclass")

    def get_huggingface_model_config(self):
        """
        :return: the huggingface pretrained config if available
        """
        raise RuntimeError(
            "get_huggingface_model_config must be implemented by subclass")

    def use_vllm_chat_completions(self):
        """
        :return: whether to use the vllm chat completions.
        """
        return False

    def get_tool_parser(self):
        """
        :return: the tool call parser if available
        """
        return None

    def get_reasoning_parser(self):
        """
        :return: the reasoning parser if available
        """
        return None

    @abstractmethod
    def inference(self, new_requests: List[Request]) -> List:
        """
        Performs prefill and decode operations for the batch.

        :param new_requests: List[Request] List of requests

        :return: generated batch decoded tokens
        """
        pass

    def add_new_requests(self, requests: List[Request]):
        """
        Adds requests to the batch when there is availability

        :param requests: List[Request] List of requests
        """
        self.active_requests.extend(requests)

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
                "last": req.is_last_token() or req.is_cancelled(),
                "content_type": req.get_content_type(),
                "request_id": req.get_client_request_id(),
            }
            if req.get_error_message():
                res["error"] = req.get_error_message()
            if req.get_error_code():
                res["code"] = req.get_error_code()
            if req.is_cancelled():
                res["error"] = res.get("error", "request has been cancelled")
                res["code"] = res.get("code", 499)
            req.reset_next_token()
            results.append(res)

        self.active_requests = [
            req for req in self.active_requests if not req.is_last_token()
        ]

        return results

    def add_lora(self,
                 lora_name: str,
                 lora_path: str,
                 long_lora_max_len: Optional[int] = None) -> bool:
        raise NotImplementedError("add_lora function not supported.")

    def remove_lora(self, lora_name) -> bool:
        raise NotImplementedError("remove_lora function not supported.")

    def pin_lora(self, lora_name) -> bool:
        raise NotImplementedError("pin_lora function not supported.")
