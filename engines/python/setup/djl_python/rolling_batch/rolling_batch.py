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
import time
from abc import ABC, abstractmethod
from typing import Union, List, Callable

from djl_python.properties_manager.properties import Properties

FINISH_REASON_MAPPER = ["length", "eos_token", "stop_sequence"]


class Token(object):
    """
        This class represents the token that comes to the output.
    """

    def __init__(self,
                 id: int,
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
        self.id = id
        self.text = text
        self.log_prob = log_prob
        self.special_token = special_token
        self.request_id = None

    def as_dict(self):
        output = {"id": self.id, "text": self.text, "log_prob": self.log_prob}
        if self.special_token:
            output["special_token"] = self.special_token
        return output

    def as_tgi_dict(self):
        output = {"id": self.id, "text": self.text, "logprob": self.log_prob}
        if self.special_token:
            output["special"] = self.special_token
        return output


def _json_output_formatter(token: Token, first_token: bool, last_token: bool,
                           details: dict, generated_text: str, id: int):
    """
    json output formatter

    :return: formatted output
    """
    json_encoded_str = f"{{\"generated_text\": \"{generated_text}" if first_token else ""
    tgi_compat = details.pop("tgi_compat", False)
    if first_token and tgi_compat:
        json_encoded_str = f"[{json_encoded_str}"
    json_encoded_str = f"{json_encoded_str}{json.dumps(token.text, ensure_ascii=False)[1:-1]}"
    if last_token:
        if details:
            final_dict = {
                "finish_reason": details.get("finish_reason", None),
                "generated_tokens": details.get("generated_tokens", None),
                "inputs": details.get("inputs", None),
                "tokens": details.get("tokens", None),
            }

            if "prompt_tokens_details" in details:
                final_dict["prefill"] = details.get("prompt_tokens_details")

            details_str = f"\"details\": {json.dumps(final_dict, ensure_ascii=False)}"
            json_encoded_str = f"{json_encoded_str}\", {details_str}}}"
        else:
            json_encoded_str = f"{json_encoded_str}\"}}"
        if tgi_compat:
            json_encoded_str = f"{json_encoded_str}]"

    return json_encoded_str


def _jsonlines_output_formatter(token: Token, first_token: bool,
                                last_token: bool, details: dict,
                                generated_text: str, id: int):
    """
    jsonlines output formatter

    :return: formatted output
    """
    tgi_compat = details.pop("tgi_compat", False)
    if tgi_compat:
        token_dict = token.as_tgi_dict()
    else:
        token_dict = token.as_dict()
    final_dict = {"token": token_dict}
    if last_token:
        final_dict["generated_text"] = generated_text
        if details:
            final_dict["details"] = {
                "finish_reason": details.get("finish_reason", None),
                "generated_tokens": details.get("generated_tokens", None),
                "inputs": details.get("inputs", None),
            }
            if "prompt_tokens_details" in details:
                final_dict["details"]["prefill"] = details.get(
                    "prompt_tokens_details")
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str


def _json_chat_output_formatter(token: Token, first_token: bool,
                                last_token: bool, details: dict,
                                generated_text: str, id: int):
    """
    json output formatter for chat completions API

    :return: formatted output
    """
    created = int(time.time())
    choice1 = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": generated_text
        }
    }
    response1 = {
        "id": f"chatcmpl-{id}",
        "object": "chat.completion",
        "created": created,
        "choices": [choice1]  # Currently only support 1 choice
    }
    json_encoded_str = f"{json.dumps(response1, ensure_ascii=False)[:-5]}" if first_token else ""
    json_encoded_str = f"{json_encoded_str}{json.dumps(token.text, ensure_ascii=False)[1:-1]}"
    if last_token:
        logprobs = None
        parameters = details.get("parameters", {})
        if parameters.get("logprobs"):
            logprobs = {
                "content": [{
                    "token":
                        t.get("text"),
                    "logprob":
                        t.get("log_prob"),
                    "bytes":
                        (b := [ord(c)
                               for c in t.get("text")] if t.get("text") else None),
                    "top_logprobs":  # Currently only support 1 top_logprobs
                        [{
                            "token": t.get("text"),
                            "logprob": t.get("log_prob"),
                            "bytes": b
                        }]
                } for t in details.get("tokens", [])
                ]
            }
        choice2 = {
            "logprobs": logprobs,
            "finish_reason": details.get("finish_reason")
        }
        prompt_tokens = int(details.get("prompt_tokens", 0))
        completion_tokens = int(details.get("generated_tokens", 0))
        response2 = {
            "choices": [choice2],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens + completion_tokens)
            }
        }
        json_encoded_str = f"{json_encoded_str}\"}}, {json.dumps(response2, ensure_ascii=False)[14:]}"
    return json_encoded_str


def _jsonlines_chat_output_formatter(token: Token, first_token: bool,
                                     last_token: bool, details: dict,
                                     generated_text: str, id: int):
    """
    jsonlines output formatter for chat completions API

    :return: formatted output
    """
    created = int(time.time())
    delta = {"content": token.text}
    if first_token:
        delta["role"] = "assistant"

    logprobs = None
    parameters = details.get("parameters", {})
    if parameters.get("logprobs"):
        logprobs = {
            "content":
                [{
                    "token": token.text,
                    "logprob": token.log_prob,
                    "bytes": (b := [ord(c) for c in token.text] if token.text else None),
                    "top_logprobs":  # Currently only support 1 top_logprobs
                        [{
                            "token": token.log_prob,
                            "logprob": token.log_prob,
                            "bytes": b
                        }]
                }]
        },
    choice = {
        "index": 0,
        "delta": delta,
        "logprobs": logprobs,
        "finish_reason": details.get("finish_reason")
    }
    response = {
        "id": f"chatcmpl-{id}",
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [choice]  # Currently only support 1 choice
    }
    json_encoded_str = json.dumps(response, ensure_ascii=False) + "\n"
    return json_encoded_str


def sse_response_formatter(*args, **kwargs):
    """
    Decorator that used to form as SSE
    """
    output_str = _jsonlines_output_formatter(*args, **kwargs)
    return f"data: {output_str}\n"


def get_output_formatter(output_formatter: Union[str, Callable], stream: bool,
                         tgi_compat: bool):
    if callable(output_formatter):
        return output_formatter, None
    if output_formatter == "json":
        return _json_output_formatter, "application/json"
    if output_formatter == "jsonlines":
        return _jsonlines_output_formatter, "application/jsonlines"
    if output_formatter == "sse":
        return sse_response_formatter, "text/event-stream"
    if output_formatter == "json_chat":
        return _json_chat_output_formatter, "application/json"
    if output_formatter == "jsonlines_chat":
        return _jsonlines_chat_output_formatter, "application/jsonlines"
    if output_formatter == "none":
        return None, "text/plain"
    if output_formatter is not None:
        # TODO: Support custom loading of user supplied output formatter
        logging.warning(f"Unsupported output formatter: {output_formatter}")
    if stream:
        if tgi_compat:
            return sse_response_formatter, "text/event-stream"
        return _jsonlines_output_formatter, "application/jsonlines"
    return _json_output_formatter, "application/json"


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


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self,
                 id: int,
                 input_text: str,
                 parameters: dict,
                 details: bool = False,
                 input_ids: list = [],
                 adapter=None,
                 output_formatter: Union[str, Callable] = None,
                 tgi_compat: bool = False):
        """
        Initialize a request

        :param id: request id
        :param input_text: request's input text
        :param parameters: list of parameters
        :param details: whether to include details
        :param input_ids: request's input ids
        :param adapter: list of adapters
        :param output_formatter: output formatter function (for example,
            _json_output_formatter, _jsonlines_output_formatter, or user provided function
        """
        self.id = id
        self.input_text = input_text
        self.parameters = parameters
        self.original_params = parameters.copy()
        self.details = details
        self.adapter = adapter
        self.input_ids = input_ids
        self.next_token_str = ""
        self.first_token = True
        self.last_token = False
        self.token_cache = None
        self.generated_tokens = []
        self.decoder_input_details = parameters.get("decoder_input_details",
                                                    False)
        self.tgi_compat = tgi_compat
        if self.details:
            self.token_cache = []
        self.full_text_prefix = input_text if parameters.pop(
            "return_full_text", False) else ""

        # output formatter
        stream = parameters.pop("stream", False)
        self.output_formatter, self.content_type = get_output_formatter(
            output_formatter, stream, self.tgi_compat)

    def __repr__(self):
        return f"<Request id: {self.id} Input {self.input_text} Parameters {self.parameters} Finished {self.last_token}>"

    def set_next_token(self,
                       next_token: Union[Token, str],
                       last_token: bool = False,
                       finish_reason: str = None,
                       prompt_tokens_details: list[dict] = None):
        """
        Sets the newly generated token.
        If the function is called for multiple times, we will append tokens to the token string.

        :param next_token: next token to be set.
        :param last_token: whether this token is the last of the sequence.
        :param finish_reason: what reason made the generation ends. Current options:
            length: end because max_output_token size reached
            eos_token: End of sequence token found
            stop_sequence: Preset stop sequence token found
        :param prompt_tokens_details: prompt tokens details when parameter decoder_input_details is true.
        """
        if isinstance(next_token, str):
            next_token = Token(-1, next_token)
        next_token.request_id = self.id
        if self.token_cache is not None:
            if self.tgi_compat:
                self.token_cache.append(next_token.as_tgi_dict())
            else:
                self.token_cache.append(next_token.as_dict())
        self.generated_tokens.append(next_token.text)
        details_dict = {}
        # making detailed information captured for each token generation
        if self.details:
            details_dict["finish_reason"] = finish_reason
            details_dict["tokens"] = self.token_cache
            details_dict["generated_tokens"] = len(self.token_cache)
            details_dict["inputs"] = self.input_text
            details_dict["parameters"] = self.original_params
            details_dict["prompt_tokens"] = len(self.input_ids)
        # Special handling for error case
        elif finish_reason == "error":
            details_dict["finish_reason"] = finish_reason
        if self.output_formatter == _json_output_formatter or self.output_formatter == sse_response_formatter:
            details_dict["tgi_compat"] = self.tgi_compat
        generated_text = self.full_text_prefix
        if last_token:
            generated_text = generated_text + ''.join(self.generated_tokens)
            if self.decoder_input_details:
                details_dict["prompt_tokens_details"] = prompt_tokens_details
        if self.output_formatter is None:
            self.next_token_str += next_token.text
        else:  # output only supports size one now
            self.next_token_str += self.output_formatter(
                next_token, self.first_token, last_token, details_dict,
                generated_text, self.id)
        self.last_token = last_token
        self.first_token = False

    def get_next_token(self) -> str:
        """
        Gets the token generated for the request.

        :return: next_token
        """
        return self.next_token_str

    def reset_next_token(self):
        """
        Reset the next token.
        """
        self.next_token_str = ""

    def is_last_token(self) -> bool:
        """
        Whether the generated token is the last one

        :return: whether last token of the sequence.
        """
        return self.last_token

    def get_content_type(self) -> str:
        """
        Content type of this particular request

        :return: content type
        """
        return self.content_type


def stop_on_any_exception(func):
    """
    Decorator that handles errors sent from backend
    """

    def try_catch_handling(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            logging.exception("Rolling batch inference error")
            for request in self.active_requests:
                token = Token(-1, "", -1, True)
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
        self.req_id_counter = 0
        self.configs = configs

    def reset(self):
        self.active_requests = []
        self.req_id_counter = 0

    def get_tokenizer(self):
        """
        :return: the tokenizer used for inference
        """
        raise RuntimeError("get_tokenizer function not supported")

    @abstractmethod
    def inference(self, input_data, parameters, adapters=None):
        """
        Performs prefill and decode operations for the batch.

        :param input_data: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :param adapters: List of adapters inputs for each request in a batch

        :return: generated batch decoded tokens
        """
        pass

    def get_new_requests(self,
                         input_data: list[str],
                         parameters: list[dict],
                         batch_size: int,
                         adapters=None) -> list[Request]:
        """
        Adds requests to the batch when there is availability

        :param input_data: (list[str]) List of input prompts.
        :param parameters: (list[str]) List of settings pertaining to each request.
        :param batch_size: (int) Maximum number of requests in a batch
        :param adapters: List of adapters inputs for each request in a batch

        :return: list of current active requests (including those that have just been added)
        """
        total_req_len = len(self.active_requests)
        if batch_size > total_req_len:
            for i in range(total_req_len, batch_size):
                data = input_data[i]
                params = parameters[i] if i < len(parameters) else {}
                adapter = adapters[i] if adapters is not None and i < len(
                    parameters) else None
                details = params.pop("details", self.configs.tgi_compat)
                request = Request(self.req_id_counter,
                                  data,
                                  params,
                                  details,
                                  input_ids=self.get_tokenizer().encode(data)
                                  if details else None,
                                  adapter=adapter,
                                  output_formatter=params.pop(
                                      "output_formatter",
                                      self.configs.output_formatter),
                                  tgi_compat=self.configs.tgi_compat)
                self.active_requests.append(request)
                self.req_id_counter += 1
        return self.active_requests[total_req_len:]

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
                "content_type": req.get_content_type()
            }
            req.reset_next_token()
            results.append(res)

        self.active_requests = [
            req for req in self.active_requests if not req.is_last_token()
        ]

        if len(self.active_requests) == 0:
            self.req_id_counter = 0

        return results
