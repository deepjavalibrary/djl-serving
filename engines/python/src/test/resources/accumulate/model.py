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
Accumulation Python model example.
"""

import logging
import numpy as np
from djl_python import Input
from djl_python import Output


class Accumulation(object):
    """
    Accumulation Model implementation.
    """
    def __init__(self):
        self.nd = np.zeros(1, dtype='float32')
        self.initialized = False

    def initialize(self, properties: dict):
        """
        Initialize model.
        """
        self.initialized = True

    def accumulate(self, inputs):
        """
        Custom service entry point function.

        :param inputs: the Input object holds a list of numpy array
        :return: the Output object to be send back
        """
        outputs = Output()
        try:
            data = inputs.get_as_numpy()
            self.nd = self.nd + data[0]
            outputs.add_as_numpy([self.nd])
        except Exception as e:
            logging.error(e, exc_info=True)
            # error handling
            outputs.set_code(500)
            outputs.set_message(str(e))

        return outputs


_service = Accumulation()


def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.accumulate(inputs)
