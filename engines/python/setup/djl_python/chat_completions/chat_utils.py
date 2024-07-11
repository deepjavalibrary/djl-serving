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
from typing import Dict

from djl_python.chat_completions.chat_properties import ChatProperties


def is_chat_completions_request(inputs: Dict) -> bool:
    return "messages" in inputs


def parse_chat_completions_request(input_map: Dict, is_rolling_batch: bool,
                                   tokenizer):
    if not is_rolling_batch:
        raise ValueError(
            "chat completions support is not currently available for dynamic batching. "
            "You must enable rolling batch to use the chat completions format."
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError(
            f"Cannot provide chat completion for tokenizer: {tokenizer.__class__}, "
            f"please ensure that your tokenizer supports chat templates.")
    chat_params = ChatProperties(**input_map)
    param = chat_params.model_dump(by_alias=True, exclude_none=True)
    messages = chat_params.messages
    images = []
    tokenizer_inputs = []
    for message in messages:
        tokenizer_inputs.append(message.get_tokenizer_inputs())
        images.extend(message.get_images())
    inputs = tokenizer.apply_chat_template(tokenizer_inputs, tokenize=False)
    param[
        "do_sample"] = chat_params.temperature is not None and chat_params.temperature > 0.0
    param["details"] = True  # Enable details for chat completions
    param[
        "output_formatter"] = "jsonlines_chat" if chat_params.stream else "json_chat"

    if images:
        param["images"] = images

    return inputs, param
