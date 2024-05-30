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
import logging
import sys
import json
import argparse

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'

sys.path.append(PYTHON_CACHE_DIR)

from djl_python.inputs import Input
from djl_python.service_loader import load_model_service


def invoke_quantization(properties):
    inputs = Input()
    inputs.properties = properties

    try:
        model_service = load_model_service(properties['model_dir'],
                                           properties['entryPoint'], None)
        model_service.invoke_handler("quantize", inputs)
    except Exception as e:
        logging.exception(f"Quantization failed {str(e)}")
        raise Exception("Quantization failed.")


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--properties', type=str, help='properties')

    args = parser.parse_args()
    properties = json.loads(args.properties)

    # DJL handler expect 'option.' is removed
    input_prop = {}
    for key, value in properties.items():
        if key.startswith("option."):
            input_prop[key[7:]] = value
        else:
            input_prop[key] = value

    invoke_quantization(input_prop)


if __name__ == '__main__':
    main()
