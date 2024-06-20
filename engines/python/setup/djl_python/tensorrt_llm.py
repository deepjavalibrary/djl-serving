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
from djl_python.tensorrt_llm_python import TRTLLMPythonService
from djl_python.utils import parse_input_with_formatter, InputFormatConfigs, rolling_batch_inference
from typing import List, Tuple


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
        self.input_format_configs = None
        self.parsed_input = None

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)

        self.rolling_batch = TRTLLMRollingBatch(
            self.trt_configs.model_id_or_path, properties, self.trt_configs)
        self.input_format_configs = InputFormatConfigs(
            is_rolling_batch=True,
            is_adapters_supported=False,
            output_formatter=self.trt_configs.output_formatter)
        self.initialized = True
        return

    # Backward compatibility.
    def parse_input(
        self, inputs: Input, tokenizer, output_formatter
    ) -> Tuple[List[str], List[int], List[dict], dict, list]:
        """
        Preprocessing function that extracts information from Input objects.

        :param output_formatter: output formatter for the request
        :param inputs :(Input) a batch of inputs, each corresponding to a new request
        :param tokenizer: the tokenizer used for inference

        :return input_data (List[str]): a list of strings, each string being the prompt in a new request
        :return input_size (List[int]): a list of ints being the size of each new request
        :return parameters (List[dict]): parameters pertaining to each request
        :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
        :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
        """
        self.parsed_input = parse_input_with_formatter(
            inputs, input_format_configs=self.input_format_configs)
        return (self.parsed_input.input_data, self.parsed_input.input_size,
                self.parsed_input.parameters, self.parsed_input.errors,
                self.parsed_input.batch)

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

        return rolling_batch_inference(self.parsed_input, inputs, outputs,
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
        properties = inputs.get_properties()
        if properties.get("rolling_batch", "disable") == "disable":
            _service = TRTLLMPythonService()
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
