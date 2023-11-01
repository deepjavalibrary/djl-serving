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
import os

from djl_python.encode_decode import decode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch

class TRTLLMService(object):

    def __init__(self):
        self.initialized = False
        self.model = None
        self.device = None
        self.tokenizer = None
        self.trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE",
                                                "FALSE").lower() == 'true'
        self.rolling_batch_type = None
        self.rolling_batch = None
        self.model_config = None

    def initialize(self, properties: dict):
        model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.rolling_batch_type = properties.get("rolling_batch", "trtllm").lower()
        if "trtllm" != self.rolling_batch_type:
            raise ValueError("only trtllm rollingbatch type supported")

        kwargs = {}

        if "output_formatter" in properties:
            kwargs["output_formatter"] = properties.get("output_formatter")
        if "waiting_steps" in properties:
            kwargs["waiting_steps"] = int(properties.get("waiting_steps"))
        if "trt_llm_model_name" in properties:
            kwargs["trt_llm_model_name"] = properties.get("trt_llm_model_name")
        if "trust_remote_code" in properties:
            self.trust_remote_code = properties.get("trust_remote_code").lower() == "true"
        kwargs["trust_remote_code"] = self.trust_remote_code
        self.rolling_batch = TRTLLMRollingBatch(model_id_or_path, self.device,
                                                properties, **kwargs)
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
                if self.rolling_batch_type:
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                else:
                    outputs.add(err, key="data", batch_index=i)
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
