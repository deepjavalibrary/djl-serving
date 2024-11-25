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
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.utils import rolling_batch_inference
from djl_python.input_parser import parse_input_with_formatter


class TRTLLMService(object):
    """
    A TRTLLMService is an intermediary for the default TensorRT-LLM handler. Its functions are invoked to turn
    Inputs into Outputs and it is responsible for sending new requests to the rolling batcher, which
    calls TensorRT-LLM in the back-end.
    """

    def __init__(self):
        self.input_format_args = None
        self.initialized = False
        self.trt_configs = None
        self.rolling_batch = None
        self.tokenizer = None

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)

        self.rolling_batch = TRTLLMRollingBatch(
            self.trt_configs.model_id_or_path, properties, self.trt_configs)
        self.tokenizer = self.rolling_batch.get_tokenizer()
        self.input_format_args = self.get_input_format_args()
        self.initialized = True
        return

    def get_input_format_args(self):
        return {
            "configs": self.trt_configs,
            "tokenizer": self.tokenizer,
            "rolling_batch": self.rolling_batch
        }

    def inference(self, inputs: Input) -> Output:
        """
        Does preprocessing and sends new requests to the rolling batch script for inference

        :param inputs: (Input) a batch of inputs, each corresponding to a new request

        :return outputs (Output): a batch of outputs that contain status code, output text, and other information
        """
        outputs = Output()

        parsed_input = parse_input_with_formatter(inputs,
                                                  **self.input_format_args)
        if parsed_input.errors and len(parsed_input.requests) == len(
                parsed_input.errors):
            for i in range(len(parsed_input.batch)):
                err = parsed_input.errors.get(i)
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            return outputs

        return rolling_batch_inference(parsed_input, inputs, outputs,
                                       self.rolling_batch)


_service = TRTLLMService()


def handle(inputs: Input) -> Output:
    """
    Handler function for the default TensorRT-LLM handler.

    :param inputs: (Input) a batch of inputs, each corresponding to a new request

    :return outputs (Output): a batch of outputs that contain status code, output text, and other information.
    """
    global _service
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
