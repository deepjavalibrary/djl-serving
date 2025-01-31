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
from typing import Dict, List, Optional, Union

from djl_python.chat_completions.vllm_chat_properties import ChatProperties
from djl_python.properties_manager.properties import Properties
from djl_python.rolling_batch.rolling_batch_vllm_utils import maybe_serialize_tool_calls
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)


def is_chat_completions_request(inputs: Dict) -> bool:
    return "messages" in inputs


def parse_chat_completions_request_vllm(
    input_map: Dict,
    is_rolling_batch: bool,
    rolling_batch,
    tokenizer,
    chat_template: Optional[str] = None,
    configs: Properties = None,
    is_mistral_tokenizer: bool = False,
):
    # Chat completions can either be a rolling batch or no-batching .
    if not (is_rolling_batch or configs.batch_size == 1):
        raise ValueError(
            "chat completions support is not currently available for dynamic batching. "
            "You must enable rolling batch to use the chat completions format."
        )

    if not is_mistral_tokenizer and not hasattr(tokenizer,
                                                "apply_chat_template"):
        raise AttributeError(
            f"Cannot provide chat completion for tokenizer: {tokenizer.__class__}, "
            f"please ensure that your tokenizer supports chat templates.")

    tool_parser = rolling_batch.get_tool_parser()
    chat_params = ChatProperties(**input_map)

    if chat_params.tool_choice == "required":
        raise ValueError("tool_choice = \"required\" is not supported!")

    if is_mistral_tokenizer:
        maybe_serialize_tool_calls(chat_params)
    elif chat_params.tool_choice == "auto" and tool_parser is None:
        raise ValueError(
            "\"auto\" tool choice requires tool_call_parser to be available")

    should_parse_tools = tool_parser is not None and (hasattr(
        chat_params, "tool_choice") and chat_params.tool_choice != "none")
    if should_parse_tools:
        chat_params = tool_parser.adjust_request(request=chat_params)

    exclude = {"messages"}
    param = chat_params.model_dump(exclude_none=True, exclude=exclude)

    tool_dicts = None if chat_params.tools is None else [
        tool.model_dump() for tool in chat_params.tools
    ]
    # TODO - figure out what we need to pass for given format
    content_format = resolve_chat_template_content_format(
        chat_template=None,
        given_format="auto",
        tokenizer=tokenizer,
    )

    conversation, mm_data = parse_chat_messages(
        chat_params.messages, rolling_batch.get_model_config(), tokenizer,
        content_format)

    prompt_data: Union[str, List[int]]
    if is_mistral_tokenizer:
        text_inputs = apply_mistral_chat_template(
            tokenizer,
            messages=chat_params.messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tools=tool_dicts,
        )
    else:
        text_inputs = apply_hf_chat_template(
            tokenizer,
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=True,
            tools=tool_dicts,
        )

    param["details"] = True  # Enable details for chat completions
    param[
        "output_formatter"] = "jsonlines_chat" if chat_params.stream else "json_chat"
    param["tool_parser"] = tool_parser
    param["chat_params"] = chat_params

    if mm_data:
        param["mm_data"] = mm_data

    # In the case of mistral, text_inputs = List[TokenIds], else = str
    return text_inputs, param
