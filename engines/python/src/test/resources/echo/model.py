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

import sys
from djl_python import Input
from djl_python import Output


def handle(inputs: Input):
    """
    Default handler function
    """
    if inputs.contains_key("exception"):
        ex = inputs.get_as_string("exception")
        raise ValueError(ex)
    elif inputs.contains_key("exit"):
        sys.exit()

    data = inputs.get_as_bytes()
    content_type = inputs.get_property("content-type")
    outputs = Output()
    outputs.add(data, key="data")
    if content_type:
        outputs.add_property("content-type", content_type)

    return outputs
