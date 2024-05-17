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
from collections import OrderedDict
from typing import Any

from lmi_dist.arg_utils import VllmEngineArgs
from vllm.outputs import CompletionOutput, RequestOutput as vLLMRequestOutput
from vllm.lora.request import LoRARequest
from djl_python.request_io import Token, RequestOutput, Sequence

from djl_python.request import Request

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}

FINISH_REASON_MAPPER = {
    "length": "length",
    "stop": "eos_token",
    "abort": "abort"
}


def update_request_cache_with_output(request_cache: OrderedDict,
                                     vllm_request_output: vLLMRequestOutput,
                                     tokenizer: Any = None) -> OrderedDict:
    request_id = vllm_request_output.request_id
    cache = request_cache[request_id]
    request_output = cache["request_output"]

    # sets  prompt token details if not set
    if vllm_request_output.prompt_logprobs and not request_output.prompt_token_details:
        for index, prompt_token_id in enumerate(
                vllm_request_output.prompt_token_ids):
            prompt_token = Token(
                id=prompt_token_id,
                text=tokenizer.decode([prompt_token_id]),
                log_prob=None if index == 0 else
                vllm_request_output.prompt_logprobs[index][prompt_token_id])
            request_output.prompt_token_details.append(prompt_token)

    # sets the details of all sequences
    update_multiple_sequences(cache, request_output, vllm_request_output)

    # remove finished requests from cache
    if vllm_request_output.finished:
        request_output.finished = True
        request_output.best_sequence_index = vllm_request_output.outputs[
            0].index
        request_cache.pop(request_id)

    return request_cache


def update_multiple_sequences(cache, request_output, vllm_request_output):
    for completion_output in vllm_request_output.outputs:

        sequence_index = completion_output.index
        if f"sequence_index_{sequence_index}" not in cache:
            cache[f"sequence_index_{sequence_index}"] = {"curr_length": 0}

        if sequence_index not in request_output.sequences:
            request_output.sequences[sequence_index] = Sequence()

        # set token of the sequence
        curr_length = cache[f"sequence_index_{sequence_index}"]["curr_length"]
        token_id = completion_output.token_ids[-1]
        text = completion_output.text[curr_length:]
        if len(text) > 0:
            token = Token(id=token_id, text=text)
            if completion_output.logprobs:
                for log_probs in completion_output.logprobs:
                    if token_id in log_probs:
                        token.log_prob = log_probs[token_id].logprob
                        break
            cache[f"sequence_index_{sequence_index}"]["curr_length"] = len(
                completion_output.text)
        else:
            token = Token(id=-1, text="")

        # set finish reason
        finish_reason = FINISH_REASON_MAPPER.get(
            completion_output.finish_reason, None)
        request_output.sequences[sequence_index].finish_reason = finish_reason
        is_last_token = finish_reason is not None
        request_output.sequences[sequence_index].set_next_token(
            token, is_last_token)


def get_speculative_decoding_metrics_record(
        completion_output: CompletionOutput,
        request_output: vLLMRequestOutput) -> dict:
    request_id = request_output.request_id
    record = {}
    record["id"] = request_id
    if len(completion_output.acceptance_history) > 0:
        record["mean_acceptance"] = 1.0 * sum(
            completion_output.acceptance_history) / len(
                completion_output.acceptance_history)
    else:
        record["mean_acceptance"] = 0
    record["prompt_size"] = len(request_output.prompt_token_ids)
    record["output_size"] = len(completion_output.token_ids)
    return record


def supports_speculative_decoding() -> bool:
    return "draft_model" in VllmEngineArgs.__annotations__


def get_lora_request_params(request: Request, lora_ids: dict) -> dict:
    result = dict()
    adapter = request.adapter
    if adapter is not None:
        adapter_name = adapter.get_property("name")
        adapter_path = adapter.get_property("src")
        adapter_id = lora_ids[adapter_name]
        result["lora_request"] = LoRARequest(adapter_name, adapter_id,
                                             adapter_path)
    return result
