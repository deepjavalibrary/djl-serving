#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Union, Callable

from djl_python.request_io import Token


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