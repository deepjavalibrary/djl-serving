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
PyTorch resnet18 model example.
"""

import logging
import time

from djl_python import Input
from djl_python import Output


class Request(object):

    def __init__(self, input_text: str, max_length, initial: bool = False):
        self.input_text = input_text
        self.max_length = max_length
        self.token_sent = 0
        self.next_token = None
        self.initial = initial

    def set_next_token(self, next_token: str):
        self.next_token = next_token
        self.initial = False

    def get_next_token(self) -> str:
        self.token_sent += 1
        return f" token_{self.input_text}_{self.token_sent}"

    def is_last_token(self) -> bool:
        return self.token_sent >= self.max_length


class RollingBatch(object):
    """
    Resnet18 Model implementation.
    """

    def __init__(self):
        self.pending_requests = []

    def inference(self, inputs):
        """
        Custom service entry point function.

        :param inputs: the Input object holds a list of numpy array
        :return: the Output object to be send back
        """
        outputs = Output()
        try:
            batch_size = inputs.get_batch_size()
            if batch_size < len(self.pending_requests):
                raise ValueError("mismatch rolling batch requests")

            batch = inputs.get_batches()
            self._merge_request(batch)
            time.sleep(0.1)

            for i in range(batch_size):
                req = self.pending_requests[i]
                res = {
                    "data": req.get_next_token(),
                    "last": req.is_last_token(),
                }
                outputs.add_as_json(res, batch_index=i)

            # remove input from pending_request if finished
            for i in range(1, batch_size + 1):
                if self.pending_requests[batch_size - i].is_last_token():
                    self.pending_requests.pop(batch_size - i)
        except Exception as e:
            logging.exception("rolling batch inference failed")
            # error handling
            outputs = Output().error(str(e))

        return outputs

    def _merge_request(self, batch):
        for i, item in enumerate(batch):
            input_map = item.get_as_json()
            data = input_map.pop("inputs", input_map)
            parameters = input_map.pop("parameters", {})
            if i >= len(self.pending_requests):
                max_length = parameters.pop("max_length", 50)
                self.pending_requests.append(
                    Request(data, max_length, initial=True))
            else:
                self.pending_requests[i].set_next_token(data)


_service = RollingBatch()


def handle(inputs: Input):
    """
    Default handler function
    """
    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
