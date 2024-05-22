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
from collections import OrderedDict
from typing import Any

from lmi_dist.arg_utils import VllmEngineArgs
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.lora.request import LoRARequest
from djl_python.request_io import Token

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
                                     request_output: RequestOutput,
                                     tokenizer: Any = None) -> OrderedDict:
    request_id = request_output.request_id
    seq_output = request_output.outputs[0]
    prev_len = request_cache[request_id]['num_generated_tokens']
    cur_len = len(seq_output.token_ids)

    new_token_ids = seq_output.token_ids[prev_len:cur_len]
    output_token_texts = []
    if hasattr(seq_output, "output_token_texts"):
        output_token_texts = seq_output.output_token_texts[prev_len:cur_len]
    if seq_output.logprobs:
        new_logprobs_list = seq_output.logprobs[prev_len:cur_len]
        new_logprobs = [
            # NOTE: vLLM 0.4.1 changed logprob type
            logprobs[token_id] if isinstance(logprobs[token_id], float) else
            logprobs[token_id].logprob
            for token_id, logprobs in zip(new_token_ids, new_logprobs_list)
        ]
    else:
        new_logprobs = [None] * len(new_token_ids)

    request_cache[request_id]["token_ids"] = new_token_ids
    request_cache[request_id]["logprobs"] = new_logprobs
    request_cache[request_id]['output_token_texts'] = output_token_texts
    request_cache[request_id][
        'cumulative_logprob'] = seq_output.cumulative_logprob
    request_cache[request_id]["text"] = seq_output.text
    request_cache[request_id]["finish_reason"] = seq_output.finish_reason
    request_cache[request_id]['num_generated_tokens'] = cur_len

    if len(request_output.outputs) > 1:
        logging.warning(
            f"Finding more than 1 output for single request {len(request_output.outputs)}"
            f"Beam search is not supported yet, use first output by default")
    request_cache[request_id]["finished"] = request_output.finished
    if "prompt_tokens_details" not in request_cache[
            request_id] and request_output.prompt_logprobs:
        request_cache[request_id]["prompt_tokens_details"] = []
        for index, prompt_token_id in enumerate(
                request_output.prompt_token_ids):
            prompt_token = Token(
                id=prompt_token_id,
                text=tokenizer.decode([prompt_token_id]),
                log_prob=None if index == 0 else
                request_output.prompt_logprobs[index][prompt_token_id])
            request_cache[request_id]["prompt_tokens_details"].append(
                prompt_token)
    return request_cache


def get_speculative_decoding_metrics_record(
        completion_output: CompletionOutput,
        request_output: RequestOutput) -> dict:
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
