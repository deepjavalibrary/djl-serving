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
from typing import Union, Callable, Dict

from typing_extensions import deprecated

from djl_python.request_io import TextGenerationOutput
from djl_python.utils import wait_till_generation_finished


def output_formatter(function):
    """
    Decorator for output_formatter. User just need to annotate @output_formatter for their custom defined function.
    :param function:  Decorator takes in the function and adds an attribute.
    :return:
    """
    # adding an attribute to the function, which is used to find the decorated function.
    function.is_output_formatter = True
    return function


def get_generated_text(sequence, request_output):
    parameters = request_output.input.parameters
    generated_text = request_output.input.input_text if parameters.get(
        "return_full_text") else ""
    for token in sequence.tokens:
        generated_text += token.text
    return generated_text


def get_sequence_details(request_output: TextGenerationOutput,
                         sequence_index: int) -> Dict:
    sequence = request_output.sequences[sequence_index]
    parameters = request_output.input.parameters

    sequence_details = {
        "finish_reason": sequence.finish_reason,
        "generated_tokens": len(sequence.tokens),
        "tokens": request_output.get_tokens_as_dict(sequence_index),
    }
    if parameters.get("decoder_input_details"):
        sequence_details["prefill"] = request_output.get_prompt_tokens_as_dict(
        )
    if parameters.get("top_n_tokens", 0) > 0:
        sequence_details["top_tokens"] = request_output.get_top_tokens_as_dict(
            sequence_index)
    return sequence_details


def _json_output_formatter_best_of(request_output: TextGenerationOutput):
    """When multiple sequences are generated, then we hold off sending the result until the generation is finished.
    This is because, in case of best_of or beam_search, we would know the best sequence only at the end of generation.
    """
    if not request_output.finished:
        return ""

    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    result = {
        "generated_text": get_generated_text(best_sequence, request_output)
    }
    details = {"inputs": request_output.input.input_text}
    details.update(
        get_sequence_details(request_output,
                             request_output.best_sequence_index))

    # other sequences indicate, all other sequences except the best/chosen sequence.
    other_sequences = []
    for index in request_output.other_sequences_indices:
        sequence = request_output.sequences[index]
        generated_text = get_generated_text(sequence, request_output)
        sequence_details = get_sequence_details(request_output, index)
        sequence_details["generated_text"] = generated_text
        other_sequences.append(sequence_details)

    if other_sequences:
        if wait_till_generation_finished(parameters):
            details["best_of_sequences"] = other_sequences
    result["details"] = details
    if request_output.input.tgi_compat:
        result = [result]
    return json.dumps(result, ensure_ascii=False)


def _json_output_formatter(request_output: TextGenerationOutput):
    """
    json output formatter

    :return: formatted output
    """

    if wait_till_generation_finished(request_output.input.parameters):
        return _json_output_formatter_best_of(request_output)

    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    # TODO: Fix this so it is not required. Right now, this call is needed to
    # advance the token iterator, which is needed for rolling batch to work properly
    next_token, _, _, is_last_token = best_sequence.get_next_token()
    if not request_output.finished:
        return ""
    details = get_details_dict(request_output, include_tokens=True)
    if details.get("finish_reason") == "error":
        final_token = best_sequence.get_last_token()
        # In non-streaming, request either succeeds or fails so do not provide the
        # partial generation response that may exist
        result = {
            "generated_text": None,
            "error": final_token.error_msg,
            "code": 400,
            "details": details,
        }
        return json.dumps(result, ensure_ascii=False)
    generated_text = get_generated_text(best_sequence, request_output)
    result = {
        "generated_text": generated_text,
    }
    if details:
        result["details"] = details
    if request_output.input.tgi_compat:
        result = [result]
    return json.dumps(result, ensure_ascii=False)


def _json_3p_output_formatter(request_output: TextGenerationOutput):
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    # TODO: Fix this so it is not required. Right now, this call is needed to
    # advance the token iterator, which is needed for rolling batch to work properly
    next_token, index, first_token, last_token = best_sequence.get_next_token()
    if not request_output.finished:
        return ""

    details_dict = get_details_dict(request_output, include_tokens=True)
    generated_text = get_generated_text(best_sequence, request_output)
    num_prompt_tokens = len(request_output.prompt_tokens_details)
    num_output_tokens = details_dict["generated_tokens"]
    finish_reason = details_dict["finish_reason"]
    body = {
        "generation": generated_text,
        "prompt_token_count": num_prompt_tokens,
        "generation_token_count": num_output_tokens,
        "stop_reason": finish_reason,
    }
    error = None
    if finish_reason == "error":
        body["generation"] = None
        body["prompt_token_count"] = 0
        body["generation_token_count"] = 0
        body["stop_reason"] = "error"
        error = {
            "error": {
                "error_code": 400,
                "error_msg": next_token.error_msg
            }
        }

    metering = {
        "inputTokenCount": num_prompt_tokens,
        "outputTokenCount": num_output_tokens,
    }
    result = {
        "body": body,
        "metering": metering,
        "content_type": "application/json",  # TODO: sort out multimodal here
    }
    if error:
        result["error"] = error
    return json.dumps(result, ensure_ascii=False)


def get_details_dict(request_output: TextGenerationOutput,
                     include_tokens: bool = True) -> Dict:
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    if parameters.get("details", request_output.input.tgi_compat):
        final_dict = {
            "finish_reason": best_sequence.finish_reason,
            "generated_tokens": len(best_sequence.tokens),
            "inputs": request_output.input.input_text,
        }

        if include_tokens:
            final_dict["tokens"] = request_output.get_tokens_as_dict()

        if parameters.get("decoder_input_details"):
            final_dict["prefill"] = request_output.get_prompt_tokens_as_dict()
        if parameters.get("top_n_tokens", 0) > 0:
            final_dict["top_tokens"] = request_output.get_top_tokens_as_dict(
                request_output.best_sequence_index)

        return final_dict
    elif best_sequence.finish_reason == "error":
        return {"finish_reason": best_sequence.finish_reason}
    else:
        return {}


def _jsonlines_output_formatter(request_output: TextGenerationOutput):
    """
    jsonlines output formatter

    :return: formatted output
    """
    tgi_compat = request_output.input.tgi_compat
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, _, _, last_token = best_sequence.get_next_token()
    # with chunked prefill, we don't generate any tokens until the full prompt has been processed.
    # that means we sometimes don't have a token to return
    if next_token is None:
        return ""
    token_dict = next_token.as_tgi_dict(
    ) if tgi_compat else next_token.as_dict()
    final_dict = {"token": token_dict}
    if last_token:
        generated_text = get_generated_text(best_sequence, request_output)
        final_dict["generated_text"] = generated_text
        details_dict = get_details_dict(request_output, include_tokens=False)
        if details_dict:
            final_dict["details"] = details_dict
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str


def _jsonlines_3p_output_formatter(request_output: TextGenerationOutput):
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, index, first_token, last_token = best_sequence.get_next_token()
    # with chunked prefill, we don't generate any tokens until the full prompt has been processed.
    # that means we sometimes don't have a token to return
    if next_token is None:
        return ""
    token_details = next_token.as_dict()
    body = {"generation": token_details["text"]}
    num_prompt_tokens = len(
        request_output.prompt_tokens_details) if first_token else None
    current_token_count = len(best_sequence.tokens)
    finish_reason = best_sequence.finish_reason if last_token else None
    body["prompt_token_count"] = num_prompt_tokens
    body["generation_token_count"] = current_token_count
    body["stop_reason"] = finish_reason
    metering = {
        "outputTokenCount": current_token_count,
    }
    if first_token:
        metering["inputTokenCount"] = num_prompt_tokens
    final_dict = {
        "body": body,
        "metering": metering,
        "content_type": "application/jsonlines"
    }
    if last_token and finish_reason == "error":
        final_dict["error"] = {
            "error_code": 400,
            "error_msg": token_details["error_msg"]
        }
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str


def _json_chat_output_formatter(request_output: TextGenerationOutput):
    """
    json output formatter for chat completions API

    :return: formatted output
    """
    parameters = request_output.input.parameters
    chat_params = parameters.get("chat_params")
    tool_parser = parameters.get("tool_parser")
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    generated_text = get_generated_text(best_sequence, request_output)
    best_sequence.get_next_token()
    if not request_output.finished:
        return ""

    created = int(time.time())
    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": generated_text,
        },
        "logprobs": None,
        "finish_reason": best_sequence.finish_reason,
    }
    if chat_params and chat_params.tool_choice and type(
            chat_params.tool_choice
    ).__name__ == "ChatCompletionNamedToolChoiceParam":
        tool_calls = [{
            "id": f"chatcmpl-tool-{id(request_output)}",
            "type": "function",
            "function": {
                "name": chat_params.tool_choice.function.name,
                "arguments": generated_text
            }
        }]
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
            },
            "tool_calls": tool_calls,
            "logprobs": None,
            "finish_reason": best_sequence.finish_reason,
        }
    elif parameters.get("tools") and (parameters.get("tool_choice") == "auto"
                                      or parameters.get("tool_choice") is None
                                      ) and parameters.get("tool_parser"):
        tool_call_info = tool_parser.extract_tool_calls(generated_text,
                                                        request=chat_params)
        auto_tools_called = tool_call_info.tools_called
        if auto_tools_called:
            tool_calls = [t.model_dump() for t in tool_call_info.tool_calls]
            choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tool_call_info.content,
                },
                "tool_calls": tool_calls,
                "logprobs": None,
                "finish_reason": "tool_calls",
            }

    if parameters.get("logprobs"):
        logprobs = {
            "content": [
                {
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
        choice["logprobs"] = logprobs

    prompt_tokens = len(request_output.prompt_tokens_details)
    completion_tokens = len(best_sequence.tokens)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (prompt_tokens + completion_tokens)
    }
    result = {
        "id": f"chatcmpl-{id(request_output)}",
        "object": "chat.completion",
        "created": created,
        "choices": [choice],
        "usage": usage,
    }
    return json.dumps(result, ensure_ascii=False)


def _jsonlines_chat_output_formatter(request_output: TextGenerationOutput):
    """
    jsonlines output formatter for chat completions API

    :return: formatted output
    """
    parameters = request_output.input.parameters
    chat_params = parameters.get("chat_params")
    tool_parser = parameters.get("tool_parser")
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, index, first_token, last_token = best_sequence.get_next_token()
    # with chunked prefill, we don't generate any tokens until the full prompt has been processed.
    # that means we sometimes don't have a token to return
    if next_token is None:
        return ""

    created = int(time.time())

    if chat_params and chat_params.tool_choice and type(
            chat_params.tool_choice
    ).__name__ == "ChatCompletionNamedToolChoiceParam":
        tool_calls = [{
            "index": 0,
            "function": {
                "name": chat_params.tool_choice.function.name,
                "arguments": next_token.text
            }
        }]
        delta = {"tool_calls": tool_calls}
    elif parameters.get("tools") and (parameters.get("tool_choice") == "auto"
                                      or parameters.get("tool_choice") is None
                                      ) and parameters.get("tool_parser"):
        current_text = get_generated_text(best_sequence, request_output)
        previous_text = current_text[0:-len(next_token.text)]
        current_token_ids = [t.id for t in best_sequence.tokens]
        previous_token_ids = current_token_ids[:-1]
        tool_call_info = tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=next_token.text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=[next_token.id],
            request=chat_params)
        if tool_call_info is None:
            return ""
        tool_calls = [
            t.model_dump(exclude_none=True) for t in tool_call_info.tool_calls
        ]
        delta = {"tool_calls": tool_calls}
    else:
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
        "id": f"chatcmpl-{id(request_output)}",
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [choice]  # Currently only support 1 choice
    }
    json_encoded_str = json.dumps(response, ensure_ascii=False) + "\n"
    return json_encoded_str


def sse_response_formatter(request_output: TextGenerationOutput):
    """
    Decorator that used to form as SSE
    """
    output_str = _jsonlines_output_formatter(request_output)
    return f"data: {output_str}\n"


@deprecated(
    "onboard to new output_formatter signature. will be removed by 0.29.0")
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
            details_dict["prompt_tokens"] = len(
                request_output.prompt_tokens_details)
        # Special handling for error case
        elif best_sequence.finish_reason == "error":
            details_dict["finish_reason"] = best_sequence.finish_reason

        next_token, index, first_token, last_token = best_sequence.get_next_token(
        )
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
    if output_formatter == "3p":
        return _json_3p_output_formatter, "application/json"
    if output_formatter == "3p_stream":
        return _jsonlines_3p_output_formatter, "application/jsonlines"
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
