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
Stateful model example.
"""
import json
import logging
import time

from djl_python import Input, Output, SessionManager


class StatefulService:

    def __init__(self):
        self.session_manager = None
        self.request_id = 0
        self.initialized = False

    def initialize(self, properties: dict):
        self.session_manager: SessionManager = SessionManager(properties)
        # load model
        self.initialized = True

    def inference(self, inputs: Input):
        data = inputs.get_as_json()
        action = data.get("action", None)
        if action == "create_session":
            return self.handle_create_session()
        elif action == "close_session":
            return self.handle_close_session(inputs)
        elif action == "regular":
            return self.regular_response(inputs)
        elif action == "streaming":
            return self.streaming_response(inputs)
        else:
            outputs = Output()
            outputs.error(f"Unknown action: {action}")
            return outputs

    def handle_create_session(self) -> Output:
        outputs = Output()
        session = self.session_manager.create_session()
        # create fixed size state
        array = session.get_as_numpy("model_state",
                                     shape=(10, 5, 5),
                                     create=True)
        array.fill(1)

        # set bytes to session
        session.put("metadata", "test")

        outputs.add_property("X-Amzn-SageMaker-Session-Id", session.session_id)
        outputs.add_property("Content-Type", "application/json")
        outputs.add_as_json({"result": "session created"})
        return outputs

    def handle_close_session(self, inputs: Input) -> Output:
        outputs = Output()
        session_id = inputs.get_property("X-Amzn-SageMaker-Session-Id")
        self.session_manager.close_session(session_id)
        outputs.add_property("X-Amzn-SageMaker-Session-Closed", "true")
        outputs.add_property("Content-Type", "application/json")
        outputs.add_as_json({"result": "session closed"})
        return outputs

    def regular_response(self, inputs: Input) -> Output:
        outputs = Output()
        outputs.add_property("Content-Type", "application/json")

        session_id = inputs.get_property("X-Amzn-SageMaker-Session-Id")
        session = self.session_manager.get_session(session_id)
        if session is None:
            outputs.error("session not found for regular request")
            return outputs

        array = session.get_as_numpy("model_state", shape=(10, 5, 5))

        outputs.add_as_json({"result": f"{array.shape}"})
        return outputs

    def streaming_response(self, inputs: Input) -> Output:
        outputs = Output()
        session_id = inputs.get_property("X-Amzn-SageMaker-Session-Id")
        session = self.session_manager.get_session(session_id)
        if session is None:
            outputs.add_property("Content-Type", "application/json")
            outputs.error("session not found for streaming request")
            return outputs

        self.request_id = session.get(".request_id", 0) + 1
        session.put(".request_id", self.request_id)
        outputs.add_property("Content-Type", self.get_content_type())
        array = session.get_as_numpy("model_state", shape=(10, 5, 5))

        outputs.add_stream_content(self.generate_stream(session, array), None)
        return outputs

    def generate_stream(self, session, array) -> str:
        for i in range(array.shape[0]):
            if session.get(".request_id") > self.request_id:
                # another request is accepted by other worker
                logging.info("request cancelled")
                yield self.format_jsonlines({"cancelled": True})
                return

            data = self.step_response(array[i])
            yield self.format_jsonlines(data)

    @staticmethod
    def step_response(state):
        time.sleep(0.5)
        return state.__str__()

    @staticmethod
    def format_jsonlines(data) -> str:
        return json.dumps(data, ensure_ascii=False) + "\n"

    @staticmethod
    def get_content_type():
        return "application/jsonlines"


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
