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
from djl_python.chat_completions.chat_properties import ChatProperties


def is_chat_completions_request(inputs: map) -> bool:
    return "messages" in inputs


def parse_chat_completions_request(inputs: map, is_rolling_batch: bool,
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
    chat_params = ChatProperties(**inputs)
    _inputs = tokenizer.apply_chat_template(chat_params.messages,
                                            tokenize=False)
    _param = chat_params.dict(exclude_unset=True,
                              exclude={
                                  'messages', 'model', 'logit_bias',
                                  'top_logprobs', 'n', 'user'
                              })
    _param["details"] = True
    _param["output_formatter"] = "jsonlines_chat" if inputs.get(
        "stream", False) else "json_chat"

    return _inputs, _param
