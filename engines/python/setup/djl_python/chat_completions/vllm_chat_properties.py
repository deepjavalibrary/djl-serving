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
from typing import Optional
from pydantic import Field
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


class ChatProperties(ChatCompletionRequest):
    """
    Chat input parameters for chat completions API.
    See https://platform.openai.com/docs/api-reference/chat/create
    """

    model: Optional[str] = Field(default=None, exclude=True)  # Unused
