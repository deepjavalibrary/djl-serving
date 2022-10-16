#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import json
import logging
import os
from typing import Optional

from djl_python.inputs import Input
from djl_python.outputs import Output
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import deepspeed


class DeepSpeedService(object):

    def __init__(self):
        self.predictor = None
        self.max_new_tokens = 0
        self.initialized = False

    def initialize(self, properties: dict):
        self.max_new_tokens = int(properties.get("max_new_tokens", "50"))
        model_dir = properties.get("model_dir")
        data_type = properties.get("data_type", "fp32")
        mp_size = int(properties.get("tensor_parallel_degree", "1"))
        model_id = properties.get("model_id")
        # LOCAL_RANK env is initialized after constructor
        device = int(os.getenv('LOCAL_RANK', '0'))
        if not model_id:
            model_id = model_dir
            config_file = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    "config.json file is required for DeepSpeed model")

            with open(config_file, "r") as f:
                config = json.load(f)
                architectures = config.get("architectures")
                if not architectures:
                    raise ValueError(
                        "No architectures found in config.json file")
                # TODO: check all supported architectures

        logging.info(
            f"Init: {model_id}, tensor_parallel_degree={mp_size}, data_type={data_type}, "
            f"device={device}, max_new_tokenx={self.max_new_tokens}")

        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if data_type == "fp16":
            model.half()

        model = deepspeed.init_inference(model,
                                         mp_size=mp_size,
                                         dtype=model.dtype,
                                         replace_method='auto',
                                         replace_with_kernel_inject=True)
        self.predictor = pipeline(task='text-generation',
                                  model=model,
                                  tokenizer=tokenizer,
                                  device=device)

        self.initialized = True

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type is not None and content_type == "application/json":
                json_input = inputs.get_as_json()
                if isinstance(json_input, dict):
                    max_tokens = json_input.pop("max_new_tokens",
                                                self.max_new_tokens)
                    data = json_input.pop("inputs", json_input)
                else:
                    max_tokens = self.max_new_tokens
                    data = json_input
            else:
                data = inputs.get_as_string()
                max_tokens = self.max_new_tokens

            result = self.predictor(data,
                                    do_sample=True,
                                    max_new_tokens=max_tokens)

            outputs = Output()
            outputs.add(result)
        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            # error handling
            outputs = Output().error(str(e))

        return outputs


_service = DeepSpeedService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
