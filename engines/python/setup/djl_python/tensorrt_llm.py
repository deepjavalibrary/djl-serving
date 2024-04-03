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

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.rolling_batch.rolling_batch import get_content_type_from_output_formatter
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.tensorrt_llm_python import TRTLLMPythonService
from djl_python.utils import parse_input


class TRTLLMService(object):
    """
    A TRTLLMService is an intermediary for the default TensorRT-LLM handler. Its functions are invoked to turn
    Inputs into Outputs and it is responsible for sending new requests to the rolling batcher, which
    calls TensorRT-LLM in the back-end.
    """

    def __init__(self):
        self.initialized = False
        self.trt_configs = None
        self.rolling_batch = None
        self.parse_input = parse_input

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)

        self.rolling_batch = TRTLLMRollingBatch(
            self.trt_configs.model_id_or_path, properties, **properties)
        self.initialized = True
        return

    def inference(self, inputs: Input) -> Output:
        """
        Does preprocessing and sends new requests to the rolling batch script for inference

        :param inputs (Input): a batch of inputs, each corresponding to a new request

        :return outputs (Output): a batch of outputs that contain status code, output text, and other information
        """
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs, self.rolling_batch.get_tokenizer(),
            self.trt_configs.output_formatter)
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

            formatter = parameters[i].get("output_formatter")
            content_type = get_content_type_from_output_formatter(formatter)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)

        return outputs


_service = TRTLLMService()


def handle(inputs: Input) -> Output:
    """
    Handler function for the default TensorRT-LLM handler.

    :param inputs: (Input) a batch of inputs, each corresponding to a new request

    :return outputs (Output): a batch of outputs that contain status code, output text, and other information.
    """
    global _service
    if not _service.initialized:
        properties = inputs.get_properties()
        if properties.get("rolling_batch", "disable") == "disable":
            _service = TRTLLMPythonService()
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
