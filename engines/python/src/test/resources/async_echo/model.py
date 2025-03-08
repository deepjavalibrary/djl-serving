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
from typing import Optional

from djl_python import Input
from djl_python import Output
from djl_python.encode_decode import decode


class AsyncEcho:

    def __init__(self):
        pass

    async def inference(self, inputs: Input) -> Output:
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
            streaming = raw_payload.pop("stream", "false").lower() == "true"

            response_text = "- unit testing async mode"
            response = {
                "data":
                json.dumps({"generate_text": input_text + response_text}),
                "last": True
            }
            # simulate inference time
            await asyncio.sleep(5)
            outputs.add(Output.binary_encode(response))
        except Exception as e:
            logging.exception("continuous batch inference failed")
            outputs = Output().error(str(e))
        logging.info(f"continuous batch inference end: {outputs}")
        return outputs


service = AsyncEcho()


async def handle(inputs: Input, cl_socket) -> Optional[Output]:
    if inputs.is_empty():
        logging.info("empty inference request")
        return None

    return await service.inference(inputs)
