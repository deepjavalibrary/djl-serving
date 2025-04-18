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
from typing import Union, Tuple
from tensorrt_llm.serve.openai_protocol import (
    ErrorResponse,
    ChatCompletionResponse,
    CompletionResponse,
)
from djl_python.async_utils import create_non_stream_output
from djl_python.outputs import Output


def trtllm_non_stream_output_formatter(
    response: Union[ErrorResponse, ChatCompletionResponse, CompletionResponse],
    **_,
) -> Output:
    if isinstance(response, ErrorResponse):
        return create_non_stream_output("",
                                        error=response.message,
                                        code=response.code)
    response_data = response.model_dump_json()
    return create_non_stream_output(response_data)


def trtllm_stream_output_formatter(
    chunk: str,
    **_,
) -> Tuple[str, bool]:
    # trtllm returns responses in sse format, 'data: {...}'
    trimmed_chunk = chunk[6:].strip()
    if trimmed_chunk == '[DONE]':
        data = ""
        last = True
    else:
        data = trimmed_chunk
        last = False
    return data, last
