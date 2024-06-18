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

from vllm import EngineArgs
from vllm.outputs import CompletionOutput, RequestOutput as vLLMRequestOutput
from vllm.lora.request import LoRARequest
from djl_python.request_io import Token, Sequence
from djl_python.request import Request
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties

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
    if vllm_request_output.prompt_logprobs and not request_output.prompt_tokens_details:
        # TODO: Temp check adding the check fo T5.
        if isinstance(vllm_request_output.prompt_token_ids, list):
            for index, prompt_token_id in enumerate(
                    vllm_request_output.prompt_token_ids):
                prompt_token = Token(
                    id=prompt_token_id,
                    text=tokenizer.decode([prompt_token_id]),
                    log_prob=None if index == 0 else vllm_request_output.
                    prompt_logprobs[index][prompt_token_id].logprob)
                request_output.prompt_tokens_details.append(prompt_token)

    # sets the details of all sequences
    update_multiple_sequences(cache, request_output, vllm_request_output)

    # remove finished requests from cache
    if vllm_request_output.finished:
        request_output.finished = True
        request_output.best_sequence_index = vllm_request_output.outputs[
            0].index
        request_cache.pop(request_id)
        for i in range(1, len(vllm_request_output.outputs)):
            index = vllm_request_output.outputs[i].index
            request_output.other_sequences_indices.append(index)

    return request_cache


def update_multiple_sequences(cache, request_output, vllm_request_output):
    for completion_output in vllm_request_output.outputs:

        sequence_index = completion_output.index
        if f"sequence_index_{sequence_index}" not in cache:
            cache[f"sequence_index_{sequence_index}"] = {
                "curr_length": 0,
                "num_generated_tokens": 0
            }

        if sequence_index not in request_output.sequences:
            request_output.sequences[sequence_index] = Sequence()

        # set token of the sequence
        # previous length of token ids generated
        prev_len = cache[f"sequence_index_{sequence_index}"][
            'num_generated_tokens']
        # curr length of the token ids generated so far
        cur_len = len(completion_output.token_ids)
        cache[f"sequence_index_{sequence_index}"][
            "num_generated_tokens"] = cur_len

        # get the newly generated token_ids
        new_token_ids = completion_output.token_ids[
            prev_len:
            cur_len] if prev_len < cur_len else completion_output.token_ids

        # get the newly generated token texts
        curr_length = cache[f"sequence_index_{sequence_index}"]["curr_length"]
        text = completion_output.text[curr_length:]
        output_token_texts = []
        if hasattr(completion_output, "output_token_texts"):
            output_token_texts = completion_output.output_token_texts[
                prev_len:
                cur_len] if prev_len < cur_len else completion_output.output_token_texts
        output_token_texts = [text] * len(
            new_token_ids) if not output_token_texts else output_token_texts

        top_tokens = []
        # calculate log probs
        if completion_output.logprobs:
            new_logprobs_list = completion_output.logprobs[
                prev_len:
                cur_len] if prev_len < cur_len else completion_output.logprobs
            new_logprobs = []
            for token_id, logprobs in zip(new_token_ids, new_logprobs_list):
                for token_id_key, logprob in logprobs.items():
                    new_logprobs.append(logprobs[token_id].logprob)
                    top_tokens.append(
                        Token(id=token_id_key,
                              text=logprob.decoded_token,
                              log_prob=logprob.logprob))

        else:
            new_logprobs = [None] * len(new_token_ids)

        # set finish reason for the generation
        finish_reason = FINISH_REASON_MAPPER.get(
            completion_output.finish_reason, None)
        request_output.sequences[sequence_index].finish_reason = finish_reason
        request_output.sequences[
            sequence_index].cumulative_log_prob = completion_output.cumulative_logprob

        if new_token_ids:
            for i, (token_id, token_text, logprob) in enumerate(
                    zip(new_token_ids, output_token_texts, new_logprobs)):
                token = Token(token_id, token_text, logprob)
                is_last_token = i == (len(new_logprobs) -
                                      1) and finish_reason is not None
                request_output.sequences[sequence_index].set_next_token(
                    token, is_last_token)
        else:
            token = Token(id=-1, text="")
            is_last_token = finish_reason is not None
            request_output.sequences[sequence_index].set_next_token(
                token, is_last_token)
            top_tokens.append(token)

        request_output.sequences[sequence_index].set_next_top_tokens(
            top_tokens)

        cache[f"sequence_index_{sequence_index}"]["curr_length"] = len(
            completion_output.text)


def get_speculative_decoding_metrics_record(
        completion_output: CompletionOutput,
        request_output: vLLMRequestOutput) -> dict:
    request_id = request_output.request_id
    record = {"id": request_id}
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
    try:
        # Moved the import inside a try to support neuron vllm container w/o lmi-dist
        from lmi_dist.arg_utils import VllmEngineArgs
        return "draft_model" in VllmEngineArgs.__annotations__
    except ImportError:
        return False


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


def get_engine_args_from_config(config: VllmRbProperties) -> EngineArgs:
    if config.device == "neuron":
        return EngineArgs(model=config.model_id_or_path,
                          preloaded_model=config.preloaded_model,
                          tensor_parallel_size=config.tensor_parallel_degree,
                          dtype=DTYPE_MAPPER[config.dtype],
                          seed=0,
                          max_model_len=config.max_model_len,
                          max_num_seqs=config.max_rolling_batch_size,
                          block_size=config.max_model_len,
                          trust_remote_code=config.trust_remote_code,
                          revision=config.revision)
    else:
        return EngineArgs(
            model=config.model_id_or_path,
            tensor_parallel_size=config.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[config.dtype],
            seed=0,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_rolling_batch_prefill_tokens,
            trust_remote_code=config.trust_remote_code,
            load_format=config.load_format,
            quantization=config.quantize,
            enable_lora=config.enable_lora,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
            lora_extra_vocab_size=config.lora_extra_vocab_size,
            max_cpu_loras=config.max_cpu_loras,
            revision=config.revision)
