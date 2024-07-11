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

import importlib
import json
import logging
import os
from importlib.machinery import SourceFileLoader
from typing import Callable, Optional


class ModelService(object):

    def __init__(self, module, model_dir):
        self.module = module
        self.model_dir = model_dir

    def invoke_handler(self, function_name, inputs):
        inputs.properties["model_dir"] = self.model_dir
        return getattr(self.module, function_name)(inputs)


def load_model_service(model_dir, entry_point, device_id):
    manifest_file = os.path.join(model_dir, "MAR-INF/MANIFEST.json")
    if not os.path.exists(manifest_file):
        if os.path.isabs(entry_point):
            if not os.path.exists(entry_point):
                raise ValueError(f"entry-point file not found {entry_point}.")
            module = SourceFileLoader("model", entry_point).load_module()
        else:
            if entry_point.endswith(".py"):
                entry_point_file = os.path.join(model_dir, entry_point)
                entry_point = entry_point[:-3]
                if not os.path.exists(entry_point_file):
                    raise ValueError(
                        f"entry-point file not found {entry_point_file}.")

            module = importlib.import_module(entry_point)

        if module is None:
            raise ValueError(
                f"Unable to load entry_point {model_dir}/{entry_point}.py")
        return ModelService(module, model_dir)

    with open("MAR-INF/MANIFEST.json") as f:
        manifest = json.load(f)

    model_name = manifest["model"]["modelName"]
    handler = manifest["model"]["handler"]
    envelope = None
    batch_size = 1
    gpu = None if device_id == "-1" else int(device_id)
    logging.info("Loading torchserve model: %s/%s", model_dir, handler)

    from ts.model_loader import ModelLoaderFactory
    from .ts_service_loader import TorchServeService

    model_loader = ModelLoaderFactory.get_model_loader()
    service = model_loader.load(model_name, model_dir, handler, gpu,
                                batch_size, envelope)

    return TorchServeService(service, model_dir)


def has_function_in_module(module, function_name):
    return hasattr(module, function_name) and callable(
        getattr(module, function_name))


def is_valid_dir(dir_path: str) -> bool:
    return os.path.exists(os.path.dirname(dir_path))


def find_decorated_function(module,
                            decorator_attribute: str) -> Optional[Callable]:
    for function_name in dir(module):
        obj = getattr(module, function_name)
        if callable(obj) and getattr(obj, decorator_attribute, False):
            return obj
    return None


def get_annotated_function(model_dir: str,
                           decorator_attribute: str) -> Optional[Callable]:
    """
    Looks for a given annotation in model.py. User have to write their function
    in model.py as of now.

    :param model_dir: model directory to look for the model.py
    :param decorator_attribute: decorated_attribute
    :return: Callable input_formatter function
    """
    try:
        service = load_model_service(model_dir, "model.py", -1)
        input_formatter_function = find_decorated_function(
            service.module, decorator_attribute)
        if input_formatter_function:
            return input_formatter_function
    except ValueError:
        # No model.py is found, we default to our default input formatter
        pass

    return None
