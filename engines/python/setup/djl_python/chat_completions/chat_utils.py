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
from typing import Dict, List, Optional, Union

from djl_python.chat_completions.chat_properties import ChatProperties
from djl_python.properties_manager.properties import Properties
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)


def is_chat_completions_request(inputs: Dict) -> bool:
    return "messages" in inputs


def parse_chat_completions_request(
    input_map: Dict,
    is_rolling_batch: bool,
    rolling_batch,
    tokenizer,
    chat_template: Optional[str] = None,
    image_token: Optional[str] = None,
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

    chat_params = ChatProperties(**input_map)
    exclude = {"messages"}
    param = chat_params.model_dump(exclude_none=True, exclude=exclude)

    conversation, mm_data = parse_chat_messages(
        chat_params.messages,
        rolling_batch.get_model_config(),
        tokenizer,
    )

    prompt_data: Union[str, List[int]]
    if is_mistral_tokenizer:
        text_inputs = apply_mistral_chat_template(
            tokenizer,
            messages=chat_params.messages,
            chat_template=chat_template,
            add_generation_prompt=True,
        )
    else:
        text_inputs = apply_hf_chat_template(
            tokenizer,
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=True,
        )

    param["details"] = True  # Enable details for chat completions
    param[
        "output_formatter"] = "jsonlines_chat" if chat_params.stream else "json_chat"

    if mm_data:
        param.update(mm_data)

    # In the case of mistral, text_inputs = List[TokenIds], else = str
    return text_inputs, param
