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

from typing_extensions import deprecated

from djl_python.request_io import Token, TextGenerationOutput, RequestOutput


def _json_output_formatter(request_output: RequestOutput):
    """
    json output formatter

    :return: formatted output
    """
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]

    parameters = request_output.input.parameters
    generated_text = ""
    if parameters.get("return_full_text"):
        generated_text = request_output.input.input_text
    next_token, first_token, last_token = best_sequence.get_next_token()
    json_encoded_str = f"{{\"generated_text\": \"{generated_text}" if first_token else ""
    tgi_compat = request_output.input.tgi_compat
    if first_token and tgi_compat:
        json_encoded_str = f"[{json_encoded_str}"
    json_encoded_str = f"{json_encoded_str}{json.dumps(next_token.text, ensure_ascii=False)[1:-1]}"
    if last_token:
        if parameters.get("details", tgi_compat):
            final_dict = {
                "finish_reason": best_sequence.finish_reason,
                "generated_tokens": len(best_sequence.tokens),
                "inputs": request_output.input.input_text,
                "tokens": request_output.get_tokens_as_dict(),
            }

            if parameters.get("decoder_input_details"):
                final_dict[
                    "prefill"] = request_output.get_prompt_tokes_as_dict()
            details_str = f"\"details\": {json.dumps(final_dict, ensure_ascii=False)}"
            json_encoded_str = f"{json_encoded_str}\", {details_str}}}"
        elif best_sequence.finish_reason == "error":
            final_dict = {"finish_reason": best_sequence.finish_reason}
            details_str = f"\"details\": {json.dumps(final_dict, ensure_ascii=False)}"
            json_encoded_str = f"{json_encoded_str}\", {details_str}}}"
        else:
            json_encoded_str = f"{json_encoded_str}\"}}"
        if tgi_compat:
            json_encoded_str = f"{json_encoded_str}]"

    return json_encoded_str


def _jsonlines_output_formatter(request_output: RequestOutput):
    """
    jsonlines output formatter

    :return: formatted output
    """
    tgi_compat = request_output.input.tgi_compat
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, _, last_token = best_sequence.get_next_token()
    token_dict = next_token.as_tgi_dict(
    ) if tgi_compat else next_token.as_dict()
    final_dict = {"token": token_dict}
    if last_token:
        generated_text = request_output.input.input_text if parameters.get(
            "return_full_text") else ""
        for token in best_sequence.tokens:
            generated_text += token.text
        final_dict["generated_text"] = generated_text
        if parameters.get("details", tgi_compat):
            final_dict["details"] = {
                "finish_reason": best_sequence.finish_reason,
                "generated_tokens": len(best_sequence.tokens),
                "inputs": request_output.input.input_text,
            }
            if parameters.get("decoder_input_details"):
                final_dict["details"][
                    "prefill"] = request_output.get_prompt_tokes_as_dict()
        elif best_sequence.finish_reason == "error":
            final_dict["details"] = {
                "finish_reason": best_sequence.finish_reason
            }
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str


def _json_chat_output_formatter(request_output: RequestOutput):
    """
    json output formatter for chat completions API

    :return: formatted output
    """
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    generated_text = request_output.input.input_text if parameters.get(
        "return_full_text") else ""
    next_token, first_token, last_token = best_sequence.get_next_token()

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
    json_encoded_str = f"{json_encoded_str}{json.dumps(next_token.text, ensure_ascii=False)[1:-1]}"
    if last_token:
        logprobs = None
        if parameters.get("logprobs"):
            logprobs = {
                "content": [{
                    "token": t.text,
                    "logprob": t.log_prob,
                    "bytes": (b := [ord(c)
                               for c in t.text] if t.text else None),
                    "top_logprobs":  # Currently only support 1 top_logprobs
                        [{
                            "token": t.text,
                            "logprob": t.log_prob,
                            "bytes": b
                        }]
                } for t in best_sequence.tokens
                ]
            }
        choice2 = {
            "logprobs": logprobs,
            "finish_reason": best_sequence.finish_reason
        }
        prompt_tokens = len(request_output.input.input_ids)
        completion_tokens = len(best_sequence.tokens)
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


def _jsonlines_chat_output_formatter(request_output: RequestOutput):
    """
    jsonlines output formatter for chat completions API

    :return: formatted output
    """
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, first_token, last_token = best_sequence.get_next_token()

    created = int(time.time())
    delta = {"content": next_token.text}
    if first_token:
        delta["role"] = "assistant"

    logprobs = None
    if parameters.get("logprobs"):
        logprobs = {
            "content":
                [{
                    "token": next_token.text,
                    "logprob": next_token.log_prob,
                    "bytes": (b := [ord(c) for c in next_token.text] if next_token.text else None),
                    "top_logprobs":  # Currently only support 1 top_logprobs
                        [{
                            "token": next_token.log_prob,
                            "logprob": next_token.log_prob,
                            "bytes": b
                        }]
                }]
        },
    choice = {
        "index": 0,
        "delta": delta,
        "logprobs": logprobs,
        "finish_reason": best_sequence.finish_reason
    }
    response = {
        "id": f"chatcmpl-{id}",
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [choice]  # Currently only support 1 choice
    }
    json_encoded_str = json.dumps(response, ensure_ascii=False) + "\n"
    return json_encoded_str


def sse_response_formatter(request_output: RequestOutput):
    """
    Decorator that used to form as SSE
    """
    output_str = _jsonlines_output_formatter(request_output)
    return f"data: {output_str}\n"


@deprecated("onboard to new output_formatter signature. will be removed by 0.29.0")
def adapt_legacy_output_formatter(request_output: TextGenerationOutput) -> str:
    sequence_index = request_output.best_sequence_index
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    parameters = request_output.input.parameters
    output_formatter = request_output.input.output_formatter
    input_text = request_output.input.input_text
    generated_text = ""
    if parameters.get("return_full_text"):
        generated_text = input_text
    next_token_str = ""
    while best_sequence.has_next_token():
        details_dict = {}
        # making detailed information captured for each token generation
        if parameters.get("details", False):
            details_dict["finish_reason"] = best_sequence.finish_reason
            details_dict["tokens"] = request_output.get_tokens_as_dict(
                sequence_index)
            details_dict["generated_tokens"] = len(best_sequence.tokens)
            details_dict["inputs"] = input_text
            details_dict["parameters"] = parameters
            details_dict["prompt_tokens"] = len(request_output.input.input_ids)
        # Special handling for error case
        elif best_sequence.finish_reason == "error":
            details_dict["finish_reason"] = best_sequence.finish_reason

        next_token, first_token, last_token = best_sequence.get_next_token()
        if last_token:
            for token in best_sequence.tokens:
                generated_text += token.text
        next_token_str += output_formatter(next_token, first_token, last_token,
                                           details_dict, generated_text,
                                           request_output.request_id)
    return next_token_str


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
