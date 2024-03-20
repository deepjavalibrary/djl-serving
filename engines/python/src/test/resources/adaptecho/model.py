#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

adapters = dict()


def register_adapter(inputs: Input):
    global adapters
    name = inputs.get_properties()["name"]
    adapters[name] = inputs
    return Output().add("Successfully registered adapter")


def unregister_adapter(inputs: Input):
    global adapters
    name = inputs.get_properties()["name"]
    del adapters[name]
    return Output().add("Successfully unregistered adapter")


def handle(inputs: Input):
    """
    Default handler function
    """

    if inputs.is_empty():
        return

    outputs = Output()

    for i, input in enumerate(inputs.get_batches()):
        data = input.get_as_string()
        if input.contains_key("adapter"):
            adapter_name = input.get_as_string("adapter")
            if adapter_name in adapters:
                adapter = adapters[adapter_name]
                option = ""
                if adapter.contains_key("echooption"):
                    option = adapter.get_as_string("echooption")
                # Registered adapter
                out = adapter_name + option + data
            else:
                # Dynamic adapter
                out = "dyn" + adapter_name + data
        else:
            out = data
        outputs.add(out, key="data", batch_index=i)

    return outputs
