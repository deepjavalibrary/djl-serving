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
import asyncio
import logging
import json
from typing import Optional, List, AsyncGenerator, Union

from djl_python import Input
from djl_python import Output
from djl_python.encode_decode import decode


async def handle_streaming_response(
        response: AsyncGenerator[str, None],
        properties: dict) -> AsyncGenerator[Output, None]:
    async for chunk in response:
        output = Output()
        for k, v in properties.items():
            output.add_property(k, v)
        if "[DONE]" in chunk:
            last = True
        else:
            last = False

        resp = {"data": chunk, "last": last}
        output.add(Output.binary_encode(resp))
        yield output


async def stream_generator(tokens: List[str]) -> AsyncGenerator[str, None]:
    for token in tokens:
        # simulate token generation inference time
        await asyncio.sleep(1)
        yield json.dumps({"token": token})


class AsyncEcho:

    def __init__(self):
        pass

    async def inference(
            self,
            inputs: Input) -> Union[Output, AsyncGenerator[Output, None]]:
        outputs = Output()
        batch = inputs.get_batches()
        first = batch[0]
        for k, v in first.get_properties().items():
            outputs.add_property(k, v)
        try:
            logging.info("continuous batch inference start")
            content_type = first.get_property("Content-Type")
            raw_payload = decode(first, content_type)
            input_text = raw_payload.pop("inputs")
            parameters = raw_payload.pop("parameters", {})
            sleep_time = int(parameters.pop("sleep_time", 5))
            streaming = raw_payload.pop("stream", "false").lower() == "true"

            response_tokens = [
                f"{input_text} unit", "testing", "async", "mode", "[DONE]"
            ]
            if streaming:
                generator = stream_generator(response_tokens)
                return handle_streaming_response(generator,
                                                 first.get_properties())

            response = {
                "data":
                json.dumps({"generated_text": " ".join(response_tokens[:4])}),
                "last":
                True
            }
            # simulate inference time
            await asyncio.sleep(sleep_time)
            outputs.add(Output.binary_encode(response))
        except Exception as e:
            logging.exception("continuous batch inference failed")
            outputs = Output().error(str(e))
        logging.info(f"continuous batch inference end: {outputs}")
        return outputs


service = AsyncEcho()


async def handle(
        inputs: Input
) -> Optional[Union[Output, AsyncGenerator[Output, None]]]:
    if inputs.is_empty():
        logging.info("empty inference request")
        return None

    return await service.inference(inputs)
