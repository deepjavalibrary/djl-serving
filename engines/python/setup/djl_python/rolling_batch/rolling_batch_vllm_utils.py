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
from typing import Any, Optional

from vllm import EngineArgs, TokensPrompt, TextPrompt
from vllm.outputs import CompletionOutput, RequestOutput as vLLMRequestOutput
from vllm.lora.request import LoRARequest

from djl_python.request_io import Token, Sequence
from djl_python.request import Request
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.utils import is_beam_search

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

    # For beam search, vllm and lmi-dist produces entirely different sequences at the same index
    # after a certain step, despite tracking previous outputs. This leads to garbage output, so we wait till
    # entire generation finishes.
    parameters = request_output.input.parameters
    if is_beam_search(parameters) and not vllm_request_output.finished:
        return request_cache

    # Prefill is complete if any of the outputs have token_ids set
    prefill_is_complete = any(
        (output.token_ids for output in vllm_request_output.outputs))
    if not prefill_is_complete:
        return request_cache

    # sets prompt token details if not set
    if not request_output.prompt_tokens_details:
        # TODO: Temp check adding the check for T5.
        if isinstance(vllm_request_output.prompt_token_ids, list):
            for index, prompt_token_id in enumerate(
                    vllm_request_output.prompt_token_ids):
                log_prob = None
                if vllm_request_output.prompt_logprobs and index > 0:
                    log_prob = vllm_request_output.prompt_logprobs[index][
                        prompt_token_id].logprob
                # TODO: this is inefficient, but it works for now. There are implications for multimodal
                # and mistral models (mistral tokenizer) when doing batch decodes.
                text = tokenizer.decode(prompt_token_id)
                prompt_token = Token(id=prompt_token_id,
                                     text=text,
                                     log_prob=log_prob)
                request_output.prompt_tokens_details.append(prompt_token)

    # sets the details of all sequences
    update_multiple_sequences(request_output, vllm_request_output)

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


def update_multiple_sequences(request_output, vllm_request_output):
    for completion_output in vllm_request_output.outputs:
        sequence_index = completion_output.index

        if sequence_index not in request_output.sequences:
            request_output.sequences[sequence_index] = Sequence()

        # get the newly generated token_ids
        new_token_ids = completion_output.token_ids

        # get the newly generated token texts for speculative decoding
        output_token_texts = []
        if hasattr(completion_output, "output_token_texts"):
            output_token_texts = completion_output.output_token_texts

        top_tokens = []
        token_texts = []
        # calculate log probs and token_texts
        if completion_output.logprobs:
            new_logprobs = []
            for token_id, logprobs in zip(new_token_ids,
                                          completion_output.logprobs):
                new_logprobs.append(logprobs[token_id].logprob)
                decoded_token = logprobs[token_id].decoded_token if logprobs[
                    token_id].decoded_token else ""
                token_texts.append(decoded_token)
                for token_id_key, logprob in logprobs.items():
                    top_tokens.append(
                        Token(id=token_id_key,
                              text=logprob.decoded_token,
                              log_prob=logprob.logprob))
        elif new_token_ids:
            # TODO: Test and remove this. logprobs is always set 1. This case should never happen.
            new_logprobs = [None] * len(new_token_ids)
            token_texts.append(completion_output.text)

        if not output_token_texts:
            if len(token_texts) != len(new_token_ids):
                raise RuntimeError(
                    f"Mismatch in the number of token_ids and its token texts generated."
                    f"new token_ids: {new_token_ids}"
                    f"new token_texts: {token_texts}")
            output_token_texts = token_texts

        # set finish reason for the generation
        finish_reason = FINISH_REASON_MAPPER.get(
            completion_output.finish_reason, None)
        request_output.sequences[sequence_index].finish_reason = finish_reason
        request_output.sequences[
            sequence_index].cumulative_log_prob = completion_output.cumulative_logprob

        if new_token_ids:
            # During last generation, length of token_texts could be lesser than new_token_ids, since the
            # last token could be a special end_token_id, for which token_text would not be returned for SD.
            new_tokens_len = min(len(new_token_ids), len(output_token_texts),
                                 len(new_logprobs))
            for i, (token_id, token_text, logprob) in enumerate(
                    zip(new_token_ids, output_token_texts, new_logprobs)):
                token = Token(token_id, token_text, logprob)
                is_last_token = i == (new_tokens_len -
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
    record["acceptance_history_len"] = len(
        completion_output.acceptance_history)
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


def create_lora_request(lora_name: str, lora_id: int, lora_path: str,
                        long_lora_max_len: Optional[int]) -> LoRARequest:
    params = {
        "lora_name": lora_name,
        "lora_int_id": lora_id,
        "lora_path": lora_path
    }
    if long_lora_max_len is not None:
        params["long_lora_max_len"] = long_lora_max_len
    return LoRARequest(**params)


def get_lora_request(lora_name: str, lora_requests: dict) -> dict:
    if lora_name not in lora_requests:
        raise ValueError(f"LoRA adapter {lora_name} not found.")
    return lora_requests[lora_name]


def get_multi_modal_data(request: Request) -> Optional[dict]:
    parameters = request.parameters
    images = parameters.pop("images", None)
    multi_modal_data = None
    if images:
        multi_modal_data = {"image": images}
    return multi_modal_data


def get_prompt_inputs(request: Request):
    text_prompt = request.request_input.input_text
    multi_modal_data = get_multi_modal_data(request)
    # TODO: In chat cases, we need to apply the chat template to the messages object to get a string
    # In both HuggingFace and mistral cases, that process can also yield token-ids directly
    # that we may want to consider passing directly to the engine
    if isinstance(text_prompt, list):
        prompt = TokensPrompt(prompt_token_ids=text_prompt)
    else:
        prompt = TextPrompt(prompt=text_prompt)

    if multi_modal_data is not None:
        prompt["multi_modal_data"] = multi_modal_data
    return prompt


def maybe_serialize_tool_calls(request):
    # Adapted from https://github.com/vllm-project/vllm/blob/v0.7.0/vllm/transformers_utils/tokenizers/mistral.py#L34-L68
    for i, message in enumerate(request.messages):
        if message.get("role") == 'assistant':
            tool_calls_validator = message.get("tool_calls", ().__iter__())
            validated_tool_calls = []
            while True:
                try:
                    tool_call = next(tool_calls_validator)  # type: ignore
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break

            request.messages[i]["tool_calls"] = validated_tool_calls
