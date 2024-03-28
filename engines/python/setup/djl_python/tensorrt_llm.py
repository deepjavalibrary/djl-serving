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

import os
import logging
import tensorrt_llm_toolkit
from tensorrt_llm_toolkit.utils import utils as toolkit_utils

from transformers import AutoConfig

from djl_python.encode_decode import encode, decode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.rolling_batch.rolling_batch import get_content_type_from_output_formatter
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request

from djl_python.properties_manager.properties import is_rolling_batch_enabled


class TRTLLMService(object):
    """
    A TRTLLMService is an intermediary for the default TensorRT-LLM handler. Its functions are invoked to turn
    Inputs into Outputs and it is responsible for sending new requests to the rolling batcher, which
    calls TensorRT-LLM in the back-end.
    """

    PYTHON_BACKEND_SUPPORTED_MODELS = {'t5'}

    def __init__(self):
        self.initialized = False
        self.trt_configs = None
        self.rolling_batch = None
        self.model = None
        self.is_rolling_batch_enabled = True

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)
        self.is_rolling_batch_enabled = is_rolling_batch_enabled(
            self.trt_configs.rolling_batch)
        self._load_model(properties)
        self.initialized = True
        return

    def parse_input(
            self, inputs: Input
    ) -> tuple[list[str], list[int], list[dict], dict, list]:
        """
        Preprocessing function that extracts information from Input objects.

        :param inputs (Input): a batch of inputs, each corresponding to a new request

        :return input_data (list[str]): a list of strings, each string being the prompt in a new request
        :return input_size (list[int]): a list of ints being the size of each new request
        :return parameters (list[dict]): parameters pertaining to each request
        :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
        :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
        """
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

            if is_chat_completions_request(input_map):
                _inputs, _param = parse_chat_completions_request(
                    input_map, True, self.rolling_batch.get_tokenizer())
            else:
                _inputs = input_map.pop("inputs", input_map)
                _param = input_map.pop("parameters", {})
                _param["stream"] = input_map.pop("stream", False)
            if not isinstance(_inputs, list):
                _inputs = [_inputs]
            input_data.extend(_inputs)
            input_size.append(len(_inputs))

            if "cached_prompt" in input_map:
                _param["cached_prompt"] = input_map.pop("cached_prompt")
            if "seed" not in _param:
                # set server provided seed if seed is not part of request
                if item.contains_key("seed"):
                    _param["seed"] = item.get_as_string(key="seed")
            if not "output_formatter" in _param:
                _param["output_formatter"] = self.trt_configs.output_formatter

            for _ in range(input_size[i]):
                parameters.append(_param)

        return input_data, input_size, parameters, errors, batch

    def _get_config(self, properties):
        model_path = self.trt_configs.model_id_or_path
        if not os.path.isfile(os.path.join(model_path, 'config.json')):
            model_path = toolkit_utils.get_python_backend_engine_path(
                model_path, properties)
            if not os.path.isfile(os.path.join(model_path, 'config.json')):
                raise ValueError(
                    f"Could not find config.json in {self.trt_configs.model_id_or_path} or"
                    f"{model_path} for TensorRT python backend")

        return AutoConfig.from_pretrained(
            model_path, trust_remote_code=self.trt_configs.trust_remote_code)

    def _load_model(self, properties):
        if self.is_rolling_batch_enabled:
            self.rolling_batch = TRTLLMRollingBatch(
                self.trt_configs.model_id_or_path, properties, **properties)
        else:
            model_config = self._get_config(properties)
            if model_config.model_type in self.PYTHON_BACKEND_SUPPORTED_MODELS:
                self.model = tensorrt_llm_toolkit.init_inference(
                    self.trt_configs.model_id_or_path,
                    **properties,
                    use_python_backend=True)
            else:
                raise ValueError(
                    f"You cannot disable rolling batch if its not any of these models"
                    f" {self.PYTHON_BACKEND_SUPPORTED_MODELS}. Please enable it with auto or tensorrt "
                    f"values to option.rolling_batch")

    def rolling_batch_inference(self, inputs: Input) -> Output:
        """
        Does preprocessing and sends new requests to the rolling batch script for inference

        :param inputs (Input): a batch of inputs, each corresponding to a new request

        :return outputs (Output): a batch of outputs that contain status code, output text, and other information
        """
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

            formatter = parameters[i].get("output_formatter")
            content_type = get_content_type_from_output_formatter(formatter)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)

        return outputs

    # TODO TrtLLM python backend: Change it once TrtLLM supports T5 with inflight batching.
    def inference(self, inputs: Input) -> Output:
        """
        Does preprocessing and sends new requests to the rolling batch script for inference

        :param inputs (Input): a batch of inputs, each corresponding to a new request

        :return outputs (Output): a batch of outputs that contain status code, output text, and other information
        """
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                outputs.add(err, key="data", batch_index=i)
            return outputs

        params = parameters[0]
        result = self.model.generate(input_data, **params)
        result = [{"generated_text": s} for s in result.batch_generation()]
        idx = 0
        for i, item in enumerate(batch):
            content_type = item.get_property("Content-Type")
            accept = item.get_property("Accept")
            if not accept:
                content_type = content_type if content_type else "application/json"
                accept = content_type if content_type.startswith(
                    "tensor/") else "application/json"
            elif "*/*" in accept:
                accept = "application/json"

            encode(outputs,
                   result[idx:idx + input_size[i]],
                   accept,
                   key=inputs.get_content().key_at(i))
            idx += input_size[i]
        return outputs


_service = TRTLLMService()


def handle(inputs: Input) -> Output:
    """
    Handler function for the default TensorRT-LLM handler.

    :param inputs (Input): a batch of inputs, each corresponding to a new request

    :return outputs (Output): a batch of outputs that contain status code, output text, and other information.
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    if _service.is_rolling_batch_enabled:
        return _service.rolling_batch_inference(inputs)
    else:
        return _service.inference(inputs)
