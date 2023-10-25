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
import numpy as np
from transformers import AutoTokenizer

from djl_python import Input, Output
import tritontoolkit

from djl_python.encode_decode import decode

model = None
tokenizer = None

def get_sample_payload(input_text, parameters, tokenizer):
    input_tokens = tokenizer([input_text])
    input_ids = input_tokens["input_ids"]
    input_ids_data = np.array(input_ids, dtype=np.int32)
    input_lengths_data = np.array([[len(input_ids)]], dtype=np.int32)
    max_new_tokens = 64
    if "max_new_tokens" in parameters.keys():
        max_new_tokens = parameters.get("max_new_tokens")
    request_output_len_data = np.array([[max_new_tokens]], dtype=np.uint32)
    streaming_data = np.array([[True]], dtype=bool)
    payload = {
        "input_ids": input_ids_data,
        "input_lengths": input_lengths_data,
        "request_output_len": request_output_len_data,
        "streaming": streaming_data,
    }
    return payload


def load_model(model_id):
    core = tritontoolkit.init_triton(model_id)
    model = core.load_model("tensorrt_llm")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def inference(inputs, model, tokenizer):
    content_type = inputs.get_property("Content-Type")
    input_map = decode(inputs, content_type)
    _inputs = input_map.pop("inputs", input_map)
    parameters = input_map.pop("parameters", {})
    if not isinstance(_inputs, list):
        _inputs = [_inputs]
    batch_size = len(_inputs)
    reqs = {}
    for i in range(batch_size):
        payload = get_sample_payload(_inputs[i], parameters, tokenizer)
        reqs[i] = model.inference_async(payload)
    output_tokens = [[] for _ in range(batch_size)]
    while len(reqs.keys()) > 0:
        completed_keys = []
        for key in reqs.keys():
            result, complete = reqs[key].get_result()
            output_tokens[key].append(result["output_ids"].squeeze().tolist())
            if complete:
                completed_keys.append(key)
        for key in completed_keys:
            reqs.pop(key)
    prediction = tokenizer.batch_decode(output_tokens,
                           skip_special_tokens=True)
    outputs = Output()
    prediction = [{"generated_text": s} for s in prediction]
    outputs.add_property("content-type", "application/json")
    outputs.add(prediction)
    return outputs

def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model_id = inputs.get_property("model_id")
        model, tokenizer = load_model(model_id)

    if inputs.is_empty():
        # Model server makes an empty call to warm up the model on startup
        return None

    return inference(inputs, model, tokenizer)
