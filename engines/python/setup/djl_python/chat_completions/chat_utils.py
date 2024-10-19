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
from typing import Dict, Optional

from djl_python.chat_completions.chat_properties import ChatProperties
from djl_python.properties_manager.properties import Properties


def is_chat_completions_request(inputs: Dict) -> bool:
    return "messages" in inputs


def parse_mistral_chat_request_inputs(messages, tokenizer):
    # TODO: get rid of this mess of an integration
    # Mistral has their own tokenizer with custom tokenization logic for chat type requests
    # This dependency is only available in vllm/lmi-dist, so we import it here as necessary
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    chat_request = ChatCompletionRequest(messages=messages)
    # The tokenized object contains the converted prompt, token ids, and images
    # Do not use the images returned by the tokenizer with vllm directly, it does not work
    tokenized = tokenizer.mistral.encode_chat_completion(chat_request)
    # when using the mistral tokenizer with a multimodal prompt, we must return the tokens
    # rather than the full tokenized text.
    # See https://github.com/vllm-project/vllm/issues/8411#issuecomment-2346505331
    return tokenized.tokens


def parse_non_mistral_chat_request_inputs(messages, tokenizer, image_token):
    tokenizer_inputs = []
    for message in messages:
        tokenizer_inputs.append(
            message.get_tokenizer_inputs(image_token=image_token))
    text_inputs = apply_chat_template(tokenizer, tokenizer_inputs)
    return text_inputs


def parse_chat_completions_request(
    input_map: Dict,
    is_rolling_batch: bool,
    tokenizer,
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

    messages = input_map.pop("messages")
    chat_params = ChatProperties(messages=messages, **input_map)
    exclude = {"messages"}
    param = chat_params.model_dump(by_alias=True,
                                   exclude_none=True,
                                   exclude=exclude)
    images = []
    for message in chat_params.messages:
        images.extend(message.get_images())

    # Less than ideal, but need a working solution for now
    # is_mistral_tokenizer can only be true if lmi-dist or vllm
    # mistral tokenization only works with these engines if we pass token ids directly, not text.
    # every other use case is designed for the actual string prompt being provided...
    if is_mistral_tokenizer:
        text_inputs = parse_mistral_chat_request_inputs(messages, tokenizer)
    else:
        text_inputs = parse_non_mistral_chat_request_inputs(
            chat_params.messages, tokenizer, image_token)

    param[
        "do_sample"] = chat_params.temperature is not None and chat_params.temperature > 0.0
    param["details"] = True  # Enable details for chat completions
    param[
        "output_formatter"] = "jsonlines_chat" if chat_params.stream else "json_chat"

    if images:
        param["images"] = images

    # In the case of mistral, text_inputs = List[TokenIds], else = str
    return text_inputs, param


def apply_chat_template(tokenizer, inputs):
    try:
        inputs = tokenizer.apply_chat_template(inputs,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return inputs
    except Exception as e:
        # add_generation_prompt doesn't work for all tokenizers...
        # This is a workaround for now until we can verify it works for all tokenizers
        return tokenizer.apply_chat_template(inputs, tokenize=False)
