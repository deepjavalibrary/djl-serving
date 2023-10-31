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

import logging
import os

import torch
from transformers import AutoConfig

from djl_python.encode_decode import decode
from djl_python.inputs import Input
from djl_python.outputs import Output


def get_torch_dtype_from_str(dtype: str):
    if dtype == "auto":
        return dtype
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "int8":
        return torch.int8
    if dtype is None:
        return None
    raise ValueError(f"Invalid data type: {dtype}")


def get_rolling_batch_class_from_str(rolling_batch_type: str, is_mpi: bool,
                                     model_config):
    from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
    return TRTLLMRollingBatch


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
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        device_id = int(properties.get("device_id", "-1"))
        self.device = f"cuda:{device_id}" if device_id >= 0 else None
        task = properties.get("task")
        tp_degree = int(properties.get("tensor_parallel_degree", "-1"))
        if "trust_remote_code" in properties:
            self.trust_remote_code = properties.get(
                "trust_remote_code").lower() == "true"
        # HF Acc handling
        kwargs = {"trust_remote_code": self.trust_remote_code}
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        if "device_map" in properties:
            kwargs["device_map"] = properties.get("device_map")
            self.device = None
            logging.info(f"Using device map {kwargs['device_map']}")
        elif tp_degree > 0 and torch.cuda.device_count() > 0:
            kwargs["device_map"] = "auto"
            self.device = None
            world_size = torch.cuda.device_count()
            assert world_size == tp_degree, f"TP degree ({tp_degree}) doesn't match available GPUs ({world_size})"
            logging.info(f"Using {world_size} gpus")

        if "data_type" in properties:
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties.get("data_type"))
        if "dtype" in properties:
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties.get("dtype"))
        if "revision" in properties:
            kwargs["revision"] = properties.get('revision')
        self.rolling_batch_type = properties.get("rolling_batch", None)

        self._read_model_config(model_id_or_path,
                                properties.get('revision', None))

        if "output_formatter" in properties:
            kwargs["output_formatter"] = properties.get("output_formatter")
        if "waiting_steps" in properties:
            kwargs["waiting_steps"] = int(properties.get("waiting_steps"))
        self.rolling_batch_type = self.rolling_batch_type.lower()
        is_mpi = properties.get("engine") != "Python"
        if is_mpi:
            self.device = int(os.getenv("LOCAL_RANK", 0))
        _rolling_batch_cls = get_rolling_batch_class_from_str(
            self.rolling_batch_type, is_mpi, self.model_config)
        self.rolling_batch = _rolling_batch_cls(model_id_or_path, self.device,
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

    def _read_model_config(self, model_config_path: str, revision=None):
        try:
            self.model_config = AutoConfig.from_pretrained(
                model_config_path,
                trust_remote_code=self.trust_remote_code,
                revision=revision)
        except Exception as e:
            logging.error(
                f"{model_config_path} does not contain a config.json. "
                f"This is required for loading huggingface models")
            raise e


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
