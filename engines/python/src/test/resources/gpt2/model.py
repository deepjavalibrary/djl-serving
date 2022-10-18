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

import os
from typing import Optional

import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

predictor = None


def get_model(properties: dict):
    model_dir = properties.get("model_dir")
    model_id = properties.get("model_id")
    mp_size = int(properties.get("tensor_parallel_degree", "2"))
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    if not model_id:
        model_id = model_dir

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = deepspeed.init_inference(model,
                                     mp_size=mp_size,
                                     dtype=model.dtype,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True)
    return pipeline(task='text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=local_rank)


def handle(inputs: Input) -> Optional[Output]:
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_string()
    output = Output()
    output.add_property("content-type", "application/json")
    result = predictor(data, do_sample=True, max_new_tokens=50)
    return output.add(result)


def test_inference():
    inputs = Input()
    inputs.properties["tensor_parallel_degree"] = "2"
    inputs.properties["data_type"] = "fp32"
    inputs.properties["device_id"] = "-1"
    inputs.properties["model_dir"] = os.getcwd()
    inputs.properties["model_id"] = "gpt2"
    text = "DeepSpeed is ".encode("utf-8")
    inputs.content.add(value=text)
    result = handle(inputs)
    print(result.content.value_at(0).decode("utf-8"))


if __name__ == '__main__':
    test_inference()
