#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Callable, Tuple, Union, List, Dict
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    CompletionResponse,
    ChatCompletionResponse,
    ErrorResponse,
    CompletionLogProbs,
)
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer

from djl_python.outputs import Output
from djl_python.async_utils import create_non_stream_output, create_stream_chunk_output


class ProcessedRequest:

    def __init__(
        self,
        vllm_request: Union[CompletionRequest, ChatCompletionRequest],
        inference_invoker: Callable,
        non_stream_output_formatter: Callable,
        stream_output_formatter: Callable,
        accumulate_chunks: bool,
        include_prompt: bool,
    ):
        self.vllm_request = vllm_request
        self.inference_invoker = inference_invoker
        # We need access to both the stream and non-stream output formatters here
        # because even with streaming requests, there may be some errors before inference that
        # result in a return of ErrorResponse object instead of AsyncGenerator
        self.non_stream_output_formatter = non_stream_output_formatter
        self.stream_output_formatter = stream_output_formatter
        self.accumulate_chunks = accumulate_chunks
        self.include_prompt = include_prompt


def convert_lmi_schema_to_completion_request(
    payload: dict, ) -> Tuple[CompletionRequest, bool, bool]:
    parameters = payload.get("parameters", {})

    completion_dict = {
        "prompt": payload.pop("inputs"),
        "max_tokens": parameters.pop("max_new_tokens", 30),
        "echo": parameters.pop("return_full_text", False),
        "truncate_prompt_tokens": parameters.pop("truncate", None),
        "n": parameters.pop("top_n_tokens", 1),
        "ignore_eos": parameters.pop("ignore_eos_token", False),
        "stream": payload.pop("stream", False),
    }
    # 1. when details are requested, return token details for the likely tokens (logprobs=1)
    # TGI only returns prompt token details when details is also enabled
    # 2. For streaming requests, echo throws an error. To maintain backwards compatibility for TGI schema,
    # we maintain a flag that the output formatter uses to know if prompt should be prepended to generated_text
    include_details_in_response = False
    include_prompt = False
    if completion_dict["stream"]:
        completion_dict["logprobs"] = 1
        completion_dict["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True
        }
        include_prompt = completion_dict.pop("echo", False)
    if parameters.pop("details", False):
        include_details_in_response = True
        completion_dict["logprobs"] = 1
        if parameters.pop("decoder_input_details", False):
            completion_dict["prompt_logprobs"] = 1
    do_sample = parameters.pop("do_sample", None)
    # when do_sample is None, just passthrough sampling params as sampling is dictated by the value of other params
    # when do_sample is False, set sampling params such that we disable sampling
    if do_sample is not None and not do_sample:
        parameters["temperature"] = 0.0

    completion_dict.update(parameters)

    return CompletionRequest(
        **completion_dict), include_details_in_response, include_prompt


def convert_completion_logprobs_to_tgi_tokens(
        completion_logprobs: CompletionLogProbs,
        tokenizer: AnyTokenizer) -> List[dict]:
    token_logprobs = completion_logprobs.token_logprobs
    tokens = completion_logprobs.tokens
    tgi_tokens = []
    for token, logprob in zip(tokens, token_logprobs):
        token_id = tokenizer.convert_tokens_to_ids(token)
        tgi_token = {"id": token_id, "text": token, "logprob": logprob}
        tgi_tokens.append(tgi_token)
    return tgi_tokens


def convert_completion_prefill_to_tgi_prefill(
        prompt_logprobs: List[Dict[int, Logprob]]) -> List[dict]:
    tgi_tokens = []
    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        token_id = next(iter(logprob_dict))
        logprob = logprob_dict[token_id]
        tgi_token = {
            "id": token_id,
            "text": logprob.decoded_token,
            "logprob": logprob.logprob,
        }
        tgi_tokens.append(tgi_token)
    return tgi_tokens


def convert_completion_response_to_lmi_schema(
    response: CompletionResponse,
    request: CompletionRequest = None,
    include_details: bool = False,
    tokenizer: AnyTokenizer = None,
) -> Output:
    primary_choice = response.choices[0]
    lmi_response = {"generated_text": primary_choice.text}
    if not include_details:
        return create_non_stream_output(lmi_response)
    details = {
        "finish_reason": primary_choice.stop_reason,
        "generated_tokens": response.usage.completion_tokens,
        "seed": request.seed,
        "prefill": [],
        "tokens": []
    }
    if primary_choice.logprobs is not None:
        details["tokens"] = convert_completion_logprobs_to_tgi_tokens(
            primary_choice.logprobs, tokenizer)
    if primary_choice.prompt_logprobs is not None:
        details["prefill"] = convert_completion_prefill_to_tgi_prefill(
            primary_choice.prompt_logprobs)

    lmi_response["details"] = details
    output = create_non_stream_output(lmi_response)
    return output


def vllm_non_stream_output_formatter(
    response: Union[ErrorResponse, ChatCompletionResponse, CompletionResponse],
    **_,
) -> Output:
    if isinstance(response, ErrorResponse):
        return create_non_stream_output("",
                                        error=response.message,
                                        code=response.code)
    response_data = response.model_dump_json()
    return create_non_stream_output(response_data)


def vllm_stream_output_formatter(
    chunk: str,
    **_,
) -> Tuple[str, bool]:
    # vllm returns responses in sse format, 'data: {...}'
    trimmed_chunk = chunk[6:].strip()
    if trimmed_chunk == '[DONE]':
        data = ""
        last = True
    else:
        data = trimmed_chunk
        last = False
    return data, last


def convert_completion_chunk_response_to_lmi_schema(
    chunk: str,
    include_details: bool = False,
    tokenizer: AnyTokenizer = None,
    history: List[str] = None,
    request: CompletionRequest = None,
    include_prompt: bool = False,
) -> Tuple[str, bool, List[str]]:
    # Vllm returns chunks in string format, and the conversion process to TGI
    # currently converts the string to an object, and then the object back to a string.
    # It's much easier to work with the object instead of manipulating the string, but inefficient
    trimmed_chunk = chunk[6:].strip()
    if trimmed_chunk == '[DONE]':
        data = ""
        return data, True, history

    vllm_completion_chunk = json.loads(trimmed_chunk)
    if "error" in vllm_completion_chunk:
        return json.dumps(vllm_completion_chunk,
                          ensure_ascii=False), True, history

    if len(vllm_completion_chunk["choices"]) == 0:
        # penultimate chunk
        return "", False, history
    choice = vllm_completion_chunk["choices"][0]
    index = choice["index"]
    token_text = choice["text"]
    history.append(token_text)
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    logprob = choice["logprobs"]["token_logprobs"][0]
    finish_reason = choice["finish_reason"]
    stop_reason = choice["stop_reason"]
    usage = vllm_completion_chunk["usage"]

    token = {
        "id": token_id,
        "text": token_text,
        "logprob": logprob,
    }
    tgi_chunk = {
        "index": index,
        "token": token,
        "generated_text": None,
        "details": None,
    }
    generation_finished = finish_reason is not None or stop_reason is not None
    if generation_finished:
        generated_text = ''.join(history)
        if include_prompt:
            generated_text = request.prompt + generated_text
        tgi_chunk["generated_text"] = generated_text
        if include_details:
            details = {
                "finish_reason": finish_reason or stop_reason,
                "seed": request.seed,
                "generated_tokens": usage["completion_tokens"] + 1,
                "input_length": usage["prompt_tokens"],
            }
            tgi_chunk["details"] = details
    json_str = json.dumps(tgi_chunk, ensure_ascii=False)
    return json_str, False, history


def lmi_with_details_non_stream_output_formatter(
        response: CompletionResponse,
        request: CompletionRequest = None,
        tokenizer: AnyTokenizer = None) -> Output:
    return convert_completion_response_to_lmi_schema(response,
                                                     include_details=True,
                                                     request=request,
                                                     tokenizer=tokenizer)


def lmi_non_stream_output_formatter(
        response: CompletionResponse,
        request: CompletionRequest = None) -> Output:
    return convert_completion_response_to_lmi_schema(response,
                                                     include_details=False,
                                                     request=request)


def lmi_with_details_stream_output_formatter(
    chunk: str,
    **kwargs,
) -> Tuple[str, bool, List[str]]:
    return convert_completion_chunk_response_to_lmi_schema(
        chunk, include_details=True, **kwargs)


def lmi_stream_output_formatter(
    chunk: str,
    **kwargs,
) -> Tuple[str, bool, List[str]]:
    return convert_completion_chunk_response_to_lmi_schema(chunk, **kwargs)
