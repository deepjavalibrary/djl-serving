#!/usr/bin/env python
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
"""
Test Python model example.
"""

import logging
import sys
import time

from djl_python import Input
from djl_python import Output


class OutOfMemoryError(Exception):
    pass

def stream_token():
    for i in range(5):
        time.sleep(1)
        yield f"t-{i}\n"


def handle(inputs: Input):
    """
    Default handler function
    """
    if inputs.contains_key("exception"):
        ex = inputs.get_as_string("exception")
        raise ValueError(ex)
    elif inputs.contains_key("typeerror"):
        return "invalid_type"
    elif inputs.contains_key("exit"):
        sys.exit()
    elif inputs.contains_key("OOM"):
        raise OutOfMemoryError

    data = inputs.get_as_bytes()

    outputs = Output()
    content_type = inputs.get_property("content-type")
    if content_type:
        outputs.add_property("content-type", content_type)

    if inputs.is_batch():
        logging.info(f"Dynamic batching size: {inputs.get_batch_size()}.")
        batch = inputs.get_batches()
        for i, item in enumerate(batch):
            outputs.add(item.get_as_bytes(), key="data", batch_index=i)
    else:
        if inputs.contains_key("stream"):
            outputs.add_stream_content(stream_token(), None)
        else:
            outputs.add(data, key="data")

    return outputs
