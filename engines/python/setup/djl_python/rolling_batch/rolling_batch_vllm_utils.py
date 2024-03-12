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
from vllm import EngineArgs
from vllm.outputs import CompletionOutput, RequestOutput

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


def update_request_cache_with_output(
        request_cache: OrderedDict,
        request_output: RequestOutput) -> OrderedDict:
    request_id = request_output.request_id
    request_cache[request_id]["id"] = request_output.outputs[0].token_ids[-1]
    request_cache[request_id]["text"] = request_output.outputs[0].text
    # calculate log_prob of the token based on the diff between two cumulative log probs
    request_cache[request_id]["log_prob"] = request_output.outputs[
        0].cumulative_logprob - request_cache[request_id]["cumulative_logprob"]
    request_cache[request_id]["cumulative_logprob"] = request_output.outputs[
        0].cumulative_logprob
    request_cache[request_id]["finish_reason"] = request_output.outputs[
        0].finish_reason
    if len(request_output.outputs) > 1:
        logging.warning(
            f"Finding more than 1 output for single request {len(request_output.outputs)}"
            f"Beam search is not supported yet, use first output by default")
    request_cache[request_id]["finished"] = request_output.finished
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
    return "draft_model" in EngineArgs.__annotations__
