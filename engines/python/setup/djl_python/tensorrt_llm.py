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

import logging

from djl_python.encode_decode import decode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties


class TRTLLMService(object):

    def __init__(self):
        self.initialized = False
        self.trt_configs = None
        self.rolling_batch = None

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)

        self.rolling_batch = TRTLLMRollingBatch(
            self.trt_configs.model_id_or_path, properties, **properties)
        self.initialized = True
        return

    def parse_input(self, inputs):
        input_data = []
        input_size = []
        parameters = []
        errors = {}
        batch = inputs.get_batches()
        for i, item in enumerate(batch):
            try:
                content_type = item.get_property("Content-Type")
                input_map = decode(item, content_type)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(f"Parse input failed: {i}")
                input_size.append(0)
                errors[i] = str(e)
                continue

            _inputs = input_map.pop("inputs", input_map)
            if not isinstance(_inputs, list):
                _inputs = [_inputs]
            input_data.extend(_inputs)
            input_size.append(len(_inputs))

            _param = input_map.pop("parameters", {})
            if "cached_prompt" in input_map:
                _param["cached_prompt"] = input_map.pop("cached_prompt")
            if not "seed" in _param:
                # set server provided seed if seed is not part of request
                if item.contains_key("seed"):
                    _param["seed"] = item.get_as_string(key="seed")
            for _ in range(input_size[i]):
                parameters.append(_param)

        return input_data, input_size, parameters, errors, batch

    def inference(self, inputs):
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            return outputs

        if inputs.get_property("reset_rollingbatch"):
            self.rolling_batch.reset()

        result = self.rolling_batch.inference(input_data, parameters)
        idx = 0
        for i in range(len(batch)):
            err = errors.get(i)
            if err:
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            else:
                outputs.add(Output.binary_encode(result[idx]),
                            key="data",
                            batch_index=i)
                idx += 1

        content_type = self.rolling_batch.get_content_type()
        if content_type:
            outputs.add_property("content-type", content_type)
        return outputs


_service = TRTLLMService()


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
