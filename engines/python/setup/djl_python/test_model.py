#!/usr/bin/env python3
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
import struct
import sys
from types import ModuleType
from typing import List, Dict, Union

from djl_python import PairList
from .arg_parser import ArgParser
from .inputs import Input
from .outputs import Output
from .np_util import to_nd_list
from .service_loader import load_model_service, ModelService


def create_request(input_files, parameters):
    request = Input()
    request.properties["device_id"] = "-1"

    if parameters:
        for parameter in parameters:
            pair = parameter.split("=", 2)
            if len(pair) != 2:
                raise ValueError(f"Invalid model parameter: {parameter}")
            request.properties[pair[0]] = pair[1]

    data_file = None
    for file in input_files:
        pair = file.split("=", 2)
        if len(pair) == 1:
            key = None
            val = pair[0]
        else:
            key = pair[0]
            val = pair[1]

        if data_file is None or key == "data":
            data_file = val

        if not os.path.exists(val):
            request.content.add(key=key, value=val.encode("utf-8"))
            if val.startswith("{") and val.endswith("}"):
                request.properties["content-type"] = "application/json"
            else:
                request.properties["content-type"] = "text/plain"
        else:
            with open(val, "rb") as f:
                request.content.add(key=key, value=f.read(-1))

    if data_file.endswith(".json"):
        request.properties["content-type"] = "application/json"
    elif data_file.endswith(".txt"):
        request.properties["content-type"] = "text/plain"
    elif data_file.endswith(".gif"):
        request.properties["content-type"] = "images/gif"
    elif data_file.endswith(".png"):
        request.properties["content-type"] = "images/png"
    elif data_file.endswith(".jpeg") or data_file.endswith(".jpg"):
        request.properties["content-type"] = "images/jpeg"
    elif data_file.endswith(".ndlist"):
        request.properties["content-type"] = "tensor/ndlist"
    elif data_file.endswith(".npz"):
        request.properties["content-type"] = "tensor/npz"

    return request


def create_concurrent_batch_request(inputs: List[Dict],
                                    properties: List[Dict] = None,
                                    serving_properties={}) -> Input:
    if properties is None:
        properties = []
    # Flatten operation properties
    flatten_properties = serving_properties
    for idx, data in enumerate(properties):
        for key, value in data.items():
            key = f"batch_{str(idx).zfill(3)}_{key}"
            flatten_properties[key] = value
    pair_list = PairList()
    # Flatten operation data field
    for idx, data in enumerate(inputs):
        key = f"batch_{str(idx).zfill(3)}_data"
        pair_list.add(key, Output._encode_json(data))

    inputs_obj = Input()
    inputs_obj.properties = flatten_properties
    inputs_obj.function_name = flatten_properties.get("handler", "handle")
    inputs_obj.content = pair_list
    flatten_properties['batch_size'] = len(inputs)
    return inputs_obj


def create_text_request(text: str, key: str = None) -> Input:
    request = Input()
    request.properties["device_id"] = "-1"
    request.properties["content-type"] = "text/plain"
    request.content.add(key=key, value=text.encode("utf-8"))
    return request


def create_numpy_request(list, key: str = None) -> Input:
    request = Input()
    request.properties["device_id"] = "-1"
    request.properties["content-type"] = "tensor/ndlist"
    request.content.add(key=key, value=to_nd_list(list))
    return request


def create_npz_request(list, key: str = None) -> Input:
    import io
    import numpy as np
    request = Input()
    request.properties["device_id"] = "-1"
    request.properties["content-type"] = "tensor/npz"
    memory_file = io.BytesIO()
    np.savez(memory_file, *list)
    memory_file.seek(0)
    request.content.add(key=key, value=memory_file.read(-1))
    return request


def extract_output_as_input(outputs: Output) -> Input:
    inputs = Input()
    inputs.properties = outputs.properties
    inputs.content = outputs.content
    return inputs


def extract_output_as_bytes(outputs: Output, key=None):
    return extract_output_as_input(outputs).get_as_bytes(key)


def extract_output_as_numpy(outputs: Output, key=None):
    return extract_output_as_input(outputs).get_as_numpy(key)


def extract_output_as_npz(outputs: Output, key=None):
    return extract_output_as_input(outputs).get_as_npz(key)


def extract_output_as_string(outputs: Output, key=None):
    return extract_output_as_input(outputs).get_as_string(key)


def retrieve_int(bytearr: bytearray, start_iter):
    end_iter = start_iter + 4
    data = bytearr[start_iter:end_iter]
    return struct.unpack(">i", data)[0], end_iter


def retrieve_short(bytearr: bytearray, start_iter):
    end_iter = start_iter + 2
    data = bytearr[start_iter:end_iter]
    return struct.unpack(">h", data)[0], end_iter


def retrieve_utf8(bytearr: bytearray, start_iter):
    length, start_iter = retrieve_short(bytearr, start_iter)
    if length < 0:
        return None
    end_iter = start_iter + length
    data = bytearr[start_iter:end_iter]
    return data.decode("utf8"), end_iter


def decode_encoded_output_binary(binary: bytearray):
    start_iter = 0
    prop_size, start_iter = retrieve_short(binary, start_iter)
    content = {}
    for _ in range(prop_size):
        key, start_iter = retrieve_utf8(binary, start_iter)
        val, start_iter = retrieve_utf8(binary, start_iter)
        content[key] = val

    return content


def load_properties(properties_dir):
    if not properties_dir:
        return {}
    properties = {}
    properties_file = os.path.join(properties_dir, 'serving.properties')
    if os.path.exists(properties_file):
        with open(properties_file, 'r') as f:
            for line in f:
                # ignoring line starting with #
                if line.startswith("#") or not line.strip():
                    continue
                key, value = line.strip().split('=', 1)
                key = key.strip()
                if key.startswith("option."):
                    key = key[7:]
                value = value.strip()
                properties[key] = value
    return properties


def update_properties_with_env_vars(kwargs):
    env_vars = os.environ
    for key, value in env_vars.items():
        if key.startswith("OPTION_"):
            key = key[7:].lower()
            if key == "entrypoint":
                key = "entryPoint"
            kwargs.setdefault(key, value)
    return kwargs


class TestHandler:

    def __init__(self,
                 entry_point: Union[str, ModuleType],
                 model_dir: str = None):
        self.serving_properties = update_properties_with_env_vars({})
        self.serving_properties.update(load_properties(model_dir))

        if isinstance(entry_point, str):
            os.chdir(model_dir)
            model_dir = os.getcwd()
            sys.path.append(model_dir)
            self.service = load_model_service(model_dir, entry_point, "-1")
        else:
            self.service = ModelService(entry_point, model_dir)

    def inference(self, inputs: Input) -> Output:
        function_name = inputs.get_function_name()
        return self.service.invoke_handler(function_name, inputs)

    def inference_batch(self,
                        inputs: List[Dict],
                        properties: List[Dict] = None,
                        serving_properties=None) -> Output:
        if serving_properties is None:
            serving_properties = self.serving_properties
        return self.inference(
            create_concurrent_batch_request(inputs, properties,
                                            serving_properties))

    def inference_rolling_batch(self,
                                inputs: List[Dict],
                                properties: List[Dict] = None,
                                serving_properties=None):
        cached_result = {}
        for idx in range(len(inputs)):
            cached_result[idx] = ""
        live_indices = [_ for _ in range(len(inputs))]
        while len(live_indices) > 0:
            outputs = self.inference_batch(inputs, properties,
                                           serving_properties)
            read_only_outputs = extract_output_as_input(outputs)
            encoded_content = read_only_outputs.get_content().values
            finished_indices = []
            for idx, binary in enumerate(encoded_content):
                data = decode_encoded_output_binary(binary)
                cached_result[live_indices[idx]] += data['data']
                if data['last'].lower() == 'true':
                    print(f"Finished request {live_indices[idx]}")
                    finished_indices.append(idx)
            for index in sorted(finished_indices, reverse=True):
                del live_indices[index]
            inputs = [{"inputs": ""} for _ in range(len(live_indices))]

        return cached_result


def run():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    args = ArgParser.test_model_args().parse_args()

    inputs = create_request(args.input, args.parameters)
    handler = TestHandler(args.entry_point, args.model_dir)
    inputs.function_name = args.handler

    outputs = handler.inference(inputs)
    print("output: " + str(outputs))


if __name__ == "__main__":
    run()
