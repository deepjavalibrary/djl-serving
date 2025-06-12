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
import json
import logging
from typing import AsyncGenerator, Callable, Optional, Union, Any

from djl_python.outputs import Output


class ProcessedRequest:

    def __init__(
        self,
        request: Any,
        inference_invoker: Callable,
        non_stream_output_formatter: Callable,
        stream_output_formatter: Callable,
        accumulate_chunks: bool,
        include_prompt: bool,
    ):
        self.request = request
        self.inference_invoker = inference_invoker
        # We need access to both the stream and non-stream output formatters here
        # because even with streaming requests, there may be some errors before inference that
        # result in a return of ErrorResponse object instead of AsyncGenerator
        self.non_stream_output_formatter = non_stream_output_formatter
        self.stream_output_formatter = stream_output_formatter
        self.accumulate_chunks = accumulate_chunks
        self.include_prompt = include_prompt


def create_non_stream_output(data: Union[str, dict],
                             error: Optional[str] = None,
                             code: Optional[int] = None) -> Output:
    return _create_output(
        data,
        True,
        "application/json",
        error=error,
        code=code,
    )


def create_stream_chunk_output(data: Union[str, dict],
                               last_chunk: bool,
                               error: Optional[str] = None,
                               code: Optional[int] = None) -> Output:
    return _create_output(
        data,
        last_chunk,
        "application/jsonlines",
        error=error,
        code=code,
    )


def _create_output(
    data: Union[str, dict],
    last_chunk: bool,
    content_type: str,
    error: Optional[str] = None,
    code: Optional[int] = None,
) -> Output:
    if isinstance(data, dict):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = data
    # Ensure newline for proper jsonlines handling. Extra newlines are fine if already in data_str
    data_str = data_str + '\n'
    response_dict = {
        "data": data_str,
        "last": last_chunk,
    }
    if error:
        response_dict["error"] = error
    if code:
        response_dict["code"] = code
    output = Output()
    output.add_property("Content-Type", content_type)
    output.add(Output.binary_encode(response_dict))
    return output


async def handle_streaming_response(
    response: AsyncGenerator[str, None],
    stream_output_formatter: Callable,
    accumulate_chunks: bool = False,
    **kwargs,
) -> AsyncGenerator[Output, None]:
    """
    This utility provides functionality that converts string outputs from one async generator
    into Output objects that can be handled by the async python engine.

    :param response: AsyncGenerator that produces strings
    :param stream_output_formatter: function that converts strings from the generator into new strings to return to the client.
    :param accumulate_chunks: whether to maintain a history of all chunks received and pass to stream_output_formatter
    :return: AsyncGenerator that produces Output objects that are returned to the model server frontend.
    """
    history = []
    async for chunk in response:
        try:
            if accumulate_chunks:
                data, last, history = stream_output_formatter(chunk,
                                                              history=history,
                                                              **kwargs)
            else:
                data, last = stream_output_formatter(chunk, **kwargs)
        except Exception as e:
            logging.exception("stream_output_formatter failed")
            output = create_stream_chunk_output("",
                                                True,
                                                error=str(e),
                                                code=424)
            yield output
            return

        output = create_stream_chunk_output(data, last)
        yield output
        if last:
            return
