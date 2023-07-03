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
# import pdb

class OutOfMemoryError(Exception):
    pass


def handle(inputs: Input):
    """
    Default handler function
    """

    if inputs.is_empty():
        raise OutOfMemoryError()

    outputs = Output()
    return outputs
