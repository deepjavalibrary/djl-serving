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
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Union, List, Callable, Optional

FINISH_REASON_MAPPER = ["length", "eos_token", "stop_sequence"]


class Token(object):
    """
        This class represents the token that comes to the output.
    """

    def __init__(self,
                 id: List[int],
                 text: str,
                 log_prob: float = None,
                 special_token: bool = None):
        """
        Initialize a Token

        :param id: token id in tokenizer
        :param text: the decoded text
        :param log_prob: log probability for the token
        :param special_token: if this token is special token
        """
        self.id = id if isinstance(id, list) else [id]
        self.text = text
        self.log_prob = log_prob
        self.special_token = special_token
        self.request_id = None

    def as_dict(self):
        output = {}
        if self.id:
            output["id"] = self.id
        if self.text:
            output["text"] = self.text
        if self.log_prob:
            output["log_prob"] = self.log_prob
        if self.special_token:
            output["special_token"] = self.special_token
        return output


def _json_output_formatter(token: Token, first_token: bool, last_token: bool,
                           details: dict, generated_tokens: str):
    """
    json output formatter

    :return: formatted output
    """
    json_encoded_str = f"{{\"generated_text\": \"{generated_tokens}" if first_token else ""
    json_encoded_str = f"{json_encoded_str}{json.dumps(token.text, ensure_ascii=False)[1:-1]}"
    if last_token:
        if details:
            details_str = f"\"details\": {json.dumps(details, ensure_ascii=False)}"
            json_encoded_str = f"{json_encoded_str}\", {details_str}}}"
        else:
            json_encoded_str = f"{json_encoded_str}\"}}"

    return json_encoded_str


def _jsonlines_output_formatter(token: Token, first_token: bool,
                                last_token: bool, details: dict,
                                generated_tokens: str):
    """
    jsonlines output formatter

    :return: formatted output
    """
    token_dict = token.as_dict()
    final_dict = {"token": token_dict}
    if last_token:
        final_dict["generated_text"] = generated_tokens
        if details:
            final_dict["details"] = {
                "finish_reason": details.get("finish_reason", None),
                "generated_tokens": details.get("generated_tokens", None),
            }
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self, id: int, input_text: str, parameters: dict):
        """
        Initialize a request

        :param id: request id
        :param input_text: request's input text
        :param parameters: list of parameters
        """
        self.id = id
        self.input_text = input_text
        self.parameters = parameters
        self.next_token_str = None
        self.first_token = True
        self.last_token = False
        self.token_cache = None
        self.generated_tokens = []
        if parameters.pop("details", False):
            self.token_cache = []
        self.full_text_prefix = input_text if parameters.pop(
            "return_full_text", False) else ""
        # spec_dec
        self.step_token_number = 0

    def __repr__(self):
        return f"<Request id: {self.id} Input {self.input_text} Parameters {self.parameters} Finished {self.last_token}>"

    def set_next_token(self,
                       next_token: Union[Token, str],
                       output_formatter: Callable,
                       last_token: bool = False,
                       finish_reason: str = None):
        """
        Sets the newly generated token.

        :param next_token: next token to be set.
        :param output_formatter: output formatter function (for example,
            _json_output_formatter, _jsonlines_output_formatter, or user provided function
        :param last_token: whether this token is the last of the sequence.
        :param finish_reason: what reason made the generation ends. Current options:
            length: end because max_output_token size reached
            eos_token: End of sequence token found
            stop_sequence: Preset stop sequence token found
        """
        if isinstance(next_token, str):
            next_token = Token([-1], next_token)
        next_token.request_id = self.id
        if self.token_cache is not None:
            self.token_cache.append(next_token.as_dict())
        self.generated_tokens.append(next_token.text)
        self.step_token_number = len(
            next_token.id) if next_token.id[0] != -1 else -1
        details = {}
        generated_text = self.full_text_prefix
        if last_token:
            generated_text = generated_text + ''.join(self.generated_tokens)
            if self.token_cache is not None:
                details["finish_reason"] = finish_reason
                details["tokens"] = self.token_cache
                details["generated_tokens"] = len(self.token_cache)
        if output_formatter is None:
            self.next_token_str = next_token.text
        else:  # output only supports size one now
            self.next_token_str = output_formatter(next_token,
                                                   self.first_token,
                                                   last_token, details,
                                                   generated_text)
        self.last_token = last_token
        self.first_token = False

    def get_next_token(self) -> str:
        """
        Gets the token generated for the request.

        :return: next_token
        """
        return self.next_token_str

    def get_step_token_number(self) -> int:
        """
        Gets the token number generated at each step

        :return: step_token_number
        """
        return self.step_token_number

    def is_last_token(self) -> bool:
        """
        Whether the generated token is the last one

        :return: whether last token of the sequence.
        """
        return self.last_token


def stop_on_any_exception(func):
    """
    Decorator that handles errors sent from backend
    """

    def try_catch_handling(self, input_data, parameters):
        try:
            return func(self, input_data, parameters)
        except Exception:
            logging.exception("Rolling batch inference error")
            err = {
                "data": "",
                "last": True,
                "step_token_num": 0,
                "code": 424,
                "error": ""
            }
            results = []
            for i in range(
                    len(self.active_requests) + len(self.pending_requests)):
                results.append(err)
            self.reset()
            return results

    return try_catch_handling


class RollingBatch(ABC):
    """
    This class initializes and maintains the SequenceBatchScheduler.
    Scheduler maintains the batch and also its search states such as past key values,
    attention masks and position ids for each decoding strategy requests.

    """

    def __init__(self, **kwargs):
        """
        Initializes the rolling batch scheduler.

        :param kwargs passed while loading the model
        """

        self.pending_requests: List[Request] = []
        self.active_requests: List[Request] = []
        self.req_id_counter = 0
        self.output_formatter = None
        self.waiting_steps = kwargs.get("waiting_steps", None)
        self.current_step = 0
        formatter = kwargs.get("output_formatter", None)

        if not formatter or "json" == formatter:
            self.output_formatter = _json_output_formatter
        elif "jsonlines" == formatter:
            self.output_formatter = _jsonlines_output_formatter
        elif "none" == formatter:
            pass
        elif callable(formatter):
            self.output_formatter = formatter
        else:
            # TODO: allows to load custom formatter from a module
            logging.warning(f"Unsupported formatter: {formatter}")

    def reset(self):
        self.pending_requests = []
        self.active_requests = []
        self.req_id_counter = 0

    @abstractmethod
    def inference(self, input_data, parameters):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch

        :return: generated batch decoded tokens
        """
        pass

    def get_new_requests(self, input_data: list[str], parameters: list[dict],
                         batch_size: int) -> list[Request]:
        """
        Adds requests to the batch when there is availability

        :param input_data (list[str]): List of input prompts.
        :param parameters (list[str]): List of settings pertaining to each request.
        :param batch_size (int): Maximum number of requests in a batch

        :return: list of current active requests (including those that have just been added)
        """
        total_req_len = len(self.active_requests) + len(self.pending_requests)
        if batch_size > total_req_len:
            for i in range(total_req_len, batch_size):
                data = input_data[i]
                params = parameters[i] if i < len(parameters) else {}
                request = Request(self.req_id_counter, data, params)
                self.pending_requests.append(request)
                self.req_id_counter += 1
        # wait steps and not feeding new requests
        if self.waiting_steps and self.current_step < self.waiting_steps:
            self.current_step += 1
            return []
        # add all pending to active requests
        active_pos = len(self.active_requests)
        self.active_requests.extend(self.pending_requests)
        # reset states
        self.pending_requests = []
        self.current_step = 0
        return self.active_requests[active_pos:]

    @abstractmethod
    def preprocess_requests(self, requests: list[Request]):
        """
        Converts requests into specific formats that are required by specific backends.

        :param requests (list[Request]): requests that will be sent to the backend after preprocessing
        """
        pass

    def postprocess_results(self) -> list[dict]:
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
                "step_token_num": req.get_step_token_number()
            }
            results.append(res)

        # add empty tokens to pending requests
        for i in range(len(self.active_requests),
                       len(self.active_requests) + len(self.pending_requests)):
            res = {"data": "", "last": False, "step_token_num": 0}
            results.append(res)

        self.active_requests = [
            req for req in self.active_requests if not req.is_last_token()
        ]

        if len(self.active_requests) + len(self.pending_requests) == 0:
            self.req_id_counter = 0

        return results

    def get_content_type(self) -> Optional[str]:
        """
        Returns information that can be added to the metadata in an Output object

        :return Optional[str]: format of output if applicable (None if user provided output formatter)
        """
        # TODO: find a way to return content-type for custom output formatter
        if self.output_formatter == _jsonlines_output_formatter:
            return "application/jsonlines"
        elif self.output_formatter == _json_output_formatter:
            return "application/json"
        return None
