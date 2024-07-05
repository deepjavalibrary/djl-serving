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
from typing import List, Dict

from djl_python.chat_completions.chat_properties import ChatProperties
from djl_python.multimodal.utils import fetch_image, get_image_text_prompt


def is_chat_completions_request(inputs: Dict) -> bool:
    return "messages" in inputs


def parse_multi_modal_chat_content(contents: List):
    prompt_texts = []
    images = []
    for content in contents:
        content_type = content.get("type")
        if content_type == "text":
            prompt_texts.append(content.get("text"))
        elif content_type == "image_url":
            image = fetch_image(content.get("image_url")["url"])
            images.append(image)
        else:
            raise ValueError("We only support types text and image_url")
    return prompt_texts, images


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
    messages = param.pop("messages")
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(message.get("content"), str):
            prompt_texts, content_images = parse_multi_modal_chat_content(
                message.get("content"))
            prompt_texts = '\n'.join(prompt_texts)
            if content_images:
                images.extend(content_images)
                content = get_image_text_prompt(prompt_texts)
            else:
                content = prompt_texts
        message["content"] = content

    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    param[
        "do_sample"] = chat_params.temperature is not None and chat_params.temperature > 0.0
    param["details"] = True  # Enable details for chat completions
    param[
        "output_formatter"] = "jsonlines_chat" if chat_params.stream else "json_chat"

    if images:
        param["images"] = images

    return inputs, param
