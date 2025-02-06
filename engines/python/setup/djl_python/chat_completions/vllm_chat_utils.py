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
from typing import Dict, List, Optional, Union, Any, Callable, Annotated, Tuple, Sequence

from pydantic import Field
from vllm import TokensPrompt
from vllm.entrypoints.openai.serving_engine import RequestPrompt, TextTokensPrompt
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.transformers_utils.tokenizers.mistral import maybe_serialize_tool_calls
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.chat_utils import (
    apply_hf_chat_template, apply_mistral_chat_template, parse_chat_messages,
    resolve_chat_template_content_format, ChatCompletionMessageParam,
    ChatTemplateContentFormatOption, ConversationMessage)

from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch

# The logic in this file is heavily inspired by https://github.com/vllm-project/vllm/blob/v0.7.1/vllm/entrypoints/openai/serving_chat.py#L109
# Many of the utilities and validation logic are modified directly from vLLM's code
# TODO: Figure out a way to integrate with vLLM at a higher level than we do now to avoid this code


def parse_chat_completions_request_vllm(
    input_map: Dict,
    rolling_batch: VLLMRollingBatch,
    tokenizer,
):

    tool_parser = rolling_batch.get_tool_parser()
    reasoning_parser = rolling_batch.get_reasoning_parser()
    model = input_map.pop("model", "lmi")
    chat_params = ChatCompletionRequest(**input_map, model=model)

    if chat_params.tool_choice == "required":
        raise ValueError("tool_choice = \"required\" is not supported!")

    if rolling_batch.is_mistral_tokenizer:
        maybe_serialize_tool_calls(chat_params)
    if (chat_params.tool_choice == "auto"
            and not (rolling_batch.vllm_configs.enable_auto_tool_choice
                     and tool_parser is not None)
            and not rolling_batch.is_mistral_tokenizer):
        raise ValueError(
            "\"auto\" tool choice requires "
            "--enable-auto-tool-choice and --tool-call-parser to be set")

    tool_dicts = None if chat_params.tools is None else [
        tool.model_dump() for tool in chat_params.tools
    ]

    conversation, request_prompt, engine_prompt, input_text = _preprocess_chat(
        chat_params,
        tokenizer,
        chat_params.messages,
        chat_params.chat_template or rolling_batch.get_chat_template(),
        rolling_batch.get_chat_template_content_format(),
        rolling_batch,
        add_generation_prompt=chat_params.add_generation_prompt,
        continue_final_message=chat_params.continue_final_message,
        tool_dicts=tool_dicts,
        documents=chat_params.documents,
        tool_parser=tool_parser,
        truncate_prompt_tokens=chat_params.truncate_prompt_tokens,
        add_special_tokens=chat_params.add_special_tokens,
    )

    default_sampling_params = rolling_batch.get_default_sampling_params()
    default_max_new_tokens = rolling_batch.engine.model_config.max_model_len - len(
        engine_prompt["prompt_token_ids"])
    sampling_params = chat_params.to_sampling_params(
        default_max_new_tokens,
        rolling_batch.engine.model_config.logits_processor_pattern,
        default_sampling_params)
    params = {
        "stream": chat_params.stream,
        "output_formatter":
        "jsonlines_chat" if chat_params.stream else "json_chat",
        "sampling_params": sampling_params,
        "conversation": conversation,
        "request_prompts": request_prompt,
        "engine_prompt": engine_prompt,
        "tool_parser": tool_parser,
        "reasoning_parser": reasoning_parser,
        "chat_params": chat_params,
    }
    return input_text, params


def _preprocess_chat(
    request: ChatCompletionRequest,
    tokenizer: AnyTokenizer,
    messages: List[ChatCompletionMessageParam],
    chat_template: Optional[str],
    chat_template_content_format: ChatTemplateContentFormatOption,
    rolling_batch: VLLMRollingBatch,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    tool_dicts: Optional[List[Dict[str, Any]]] = None,
    documents: Optional[List[Dict[str, str]]] = None,
    tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None,
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
    add_special_tokens: bool = False,
) -> Tuple[List[ConversationMessage], RequestPrompt, TokensPrompt, str]:
    resolved_content_format = resolve_chat_template_content_format(
        chat_template, chat_template_content_format, tokenizer)
    conversation, mm_data = parse_chat_messages(
        messages,
        rolling_batch.engine.model_config,
        tokenizer,
        content_format=resolved_content_format,
    )
    chat_template_kwargs: Dict[str, Any] = dict(
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tools=tool_dicts,
        documents=documents,
    )

    request_prompt: Union[str, List[int]]
    if rolling_batch.is_mistral_tokenizer:
        request_prompt = apply_mistral_chat_template(tokenizer,
                                                     messages=messages,
                                                     **chat_template_kwargs)
    else:
        request_prompt = apply_hf_chat_template(tokenizer,
                                                conversation=conversation,
                                                **chat_template_kwargs)

    should_parse_tools = tool_parser is not None and request.tool_choice != "none"
    if should_parse_tools:
        request = tool_parser.adjust_request(request=request)

    if isinstance(request_prompt, str):
        # Hf tokenizer case
        prompt_inputs = tokenize_prompt_input(
            request,
            tokenizer,
            request_prompt,
            rolling_batch.engine.model_config.max_model_len,
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens,
        )
    else:
        # MistralTokenizer case
        prompt_inputs = TextTokensPrompt(
            prompt=tokenizer.decode(request_prompt),
            prompt_token_ids=request_prompt)

    engine_prompt = TokensPrompt(
        prompt_token_ids=prompt_inputs["prompt_token_ids"])
    if mm_data is not None:
        engine_prompt["multi_modal_data"] = mm_data
    return conversation, request_prompt, engine_prompt, prompt_inputs["prompt"]


def tokenize_prompt_input(request: ChatCompletionRequest,
                          tokenizer: AnyTokenizer,
                          prompt_input: Union[str, List[int]],
                          max_model_len: int,
                          truncate_prompt_tokens: Optional[Annotated[
                              int, Field(ge=1)]] = None,
                          add_special_tokens: bool = True) -> TextTokensPrompt:
    if isinstance(prompt_input, str):
        return normalize_prompt_text_to_input(
            request,
            tokenizer,
            prompt_input,
            truncate_prompt_tokens,
            add_special_tokens,
            max_model_len,
        )
    else:
        return normalize_prompt_tokens_to_input(
            request,
            tokenizer,
            prompt_input,
            truncate_prompt_tokens,
            max_model_len,
        )


def normalize_prompt_text_to_input(
    request: ChatCompletionRequest,
    tokenizer: AnyTokenizer,
    prompt: str,
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    add_special_tokens: bool,
    max_model_len: int,
) -> TextTokensPrompt:
    if truncate_prompt_tokens is None:
        encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
    else:
        encoded = tokenizer(prompt,
                            add_special_tokens=add_special_tokens,
                            truncation=True,
                            max_length=truncate_prompt_tokens)

    return validate_input(request, encoded.input_ids, prompt, max_model_len)


def normalize_prompt_tokens_to_input(
    request: ChatCompletionRequest,
    tokenizer: AnyTokenizer,
    prompt_ids: List[int],
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    max_model_len: int,
) -> TextTokensPrompt:
    if truncate_prompt_tokens is None:
        input_ids = prompt_ids
    else:
        input_ids = prompt_ids[-truncate_prompt_tokens:]
    input_text = tokenizer.decode(input_ids)
    return validate_input(request, input_ids, input_text, max_model_len)


def validate_input(
    request: ChatCompletionRequest,
    input_ids: List[int],
    input_text: str,
    max_model_len: int,
) -> TextTokensPrompt:
    token_num = len(input_ids)

    # chat completion endpoint supports max_completion_tokens
    max_tokens = request.max_completion_tokens or request.max_tokens
    if max_tokens is None:
        if token_num >= max_model_len:
            raise ValueError(f"This model's maximum context length is "
                             f"{max_model_len} tokens. However, you requested "
                             f"{token_num} tokens in the messages, "
                             f"Please reduce the length of the messages.")
    elif token_num + max_tokens > max_model_len:
        raise ValueError(
            f"This model's maximum context length is "
            f"{max_model_len} tokens. However, you requested "
            f"{max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.")

    return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)
