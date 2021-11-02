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
import sys

from .arg_parser import ArgParser
from .inputs import Input
from .service_loader import load_model_service


def create_request(input_files):
    request = Input()
    request.properties["device_id"] = "-1"

    data_file = None
    for file in input_files:
        pair = file.split("=")
        if len(pair) == 1:
            key = None
            val = pair[0]
        else:
            key = pair[0]
            val = pair[1]

        if data_file is None or key == "data":
            data_file = val

        if not os.path.exists(val):
            raise ValueError("--input file not found {}.".format(val))

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
    elif data_file.endswith(".ndlist") or data_file.endswith(
            ".npy") or data_file.endswith(".npz"):
        request.properties["content-type"] = "tensor/ndlist"

    return request


def run():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    args = ArgParser.test_model_args().parse_args()

    inputs = create_request(args.input)
    inputs.function_name = args.handler

    os.chdir(args.model_dir)
    model_dir = os.getcwd()
    sys.path.append(model_dir)

    entry_point = args.entry_point
    service = load_model_service(model_dir, entry_point, "-1")

    function_name = inputs.get_function_name()
    outputs = service.invoke_handler(function_name, inputs)
    print("output: " + str(outputs))


if __name__ == "__main__":
    run()
