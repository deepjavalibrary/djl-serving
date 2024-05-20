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

from djl_python import Input, Output, SessionManager, get_session_manager


class StatefulService:

    def __init__(self):
        self.sessions = None
        self.initialized = False

    def initialize(self, properties: dict):
        self.sessions: SessionManager = get_session_manager(properties)
        self.initialized = True

    def inference(self, inputs: Input):
        outputs = Output()

        for i, input in enumerate(inputs.get_batches()):
            data = input.get_as_string()
            if input.contains_key("session_id"):
                session_id = input.get_as_string("session_id")
                mem, new_mem = self.sessions.load(session_id)
                if new_mem:
                    current_value = 1
                else:
                    current_value = mem.read_byte() + 1
                    mem.seek(0)
                out = str(current_value) + data
                mem.write_byte(current_value)
                self.sessions.save(session_id)
            else:
                out = data
            outputs.add(out, key="data", batch_index=i)

        return outputs


_service = StatefulService()


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

    return _service.inference(inputs)
