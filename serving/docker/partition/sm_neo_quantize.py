#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import sys
import logging
import os
from types import SimpleNamespace
from typing import Final
import torch
import json

from sm_neo_utils import (OptimizationFatalError, InputConfiguration,
                          write_error_to_file, get_neo_env_vars,
                          update_dataset_cache_location)
from utils import (extract_python_jar, load_properties)
from properties_manager import PropertiesManager
from partition import PartitionService

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'
AUTOFP8_CONFIG_ENVVAR = 'AUTOFP8_CONFIG'


class NeoQuantizationService():

    def __init__(self):
        self.args: SimpleNamespace = SimpleNamespace()
        self.properties_manager: PropertiesManager = None
        self.compiler_flags: dict = None

        neo_environ = get_neo_env_vars()
        self.INPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_INPUT_MODEL_DIR"]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_COMPILED_MODEL_DIR"]
        self.COMPILATION_ERROR_FILE: Final[str] = neo_environ[
            "SM_NEO_COMPILATION_ERROR_FILE"]
        self.HF_CACHE_LOCATION: Final[str] = neo_environ["SM_NEO_HF_CACHE_DIR"]

        self.customer_properties: dict = load_properties(
            self.INPUT_MODEL_DIRECTORY)
        self.autofp8_config = None

    def initialize_partition_args_namespace(self):
        """
        Initialize args, a SimpleNamespace object that is used to instantiate a
        PropertiesManager for partitioning. PropertiesManager expects an
        argparse.Namespace, but we use a SimpleNamespace in its place because it
        is easier to construct.

        These attributes are defined in the partition.py argparser.
        PropertiesManager expects these attributes to be defined to be initialized.
        """
        self.args.save_mp_checkpoint_path = self.OUTPUT_MODEL_DIRECTORY
        num_gpus = torch.cuda.device_count()
        self.args.tensor_parallel_degree = num_gpus
        self.args.properties_dir = self.INPUT_MODEL_DIRECTORY
        self.args.quantize = None
        if not os.environ.get(
                'OPTION_QUANTIZE') and not self.customer_properties.get(
                    "option.quantize"):
            self.args.quantize = 'awq'
        self.args.pipeline_parallel_degree = None
        self.args.model_id = None
        self.args.skip_copy = None
        self.args.engine = None

    def construct_properties_manager(self):
        """
        Factory method used to construct a QuantizationPropertiesManager from
        given serving.properties
        """
        logging.debug("Constructing PropertiesManager from "
                      f"serving.properties\nargs:{self.args}\n")

        # Always quantize with device_map=auto to avoid errors caused by tensors loaded on different devices
        addl_properties = {"option.device_map": "auto"}
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=addl_properties)

    def parse_autofp8_config(self) -> dict:
        autofp8_config = os.environ.get(AUTOFP8_CONFIG_ENVVAR, {})
        if autofp8_config:
            try:
                autofp8_config = json.loads(autofp8_config)
                if not isinstance(autofp8_config, dict):
                    raise ValueError("Parsed JSON is not a dictionary")
                self.autofp8_config = autofp8_config
            except Exception as exc:
                raise InputConfiguration(
                    f"Failed to parse AutoFP8 configuration: {exc}")

    def run_quantization(self) -> str:
        """
        :return: the output of the partition command captured from stdout
        """
        partition_service = PartitionService(self.properties_manager)
        extract_python_jar(PYTHON_CACHE_DIR)
        try:
            return partition_service.run_quantization(self.autofp8_config)
        except Exception as exc:
            raise OptimizationFatalError(
                f"Encountered an error during quantization: {exc}")

    def write_properties(self):
        """
        Updates outputted serving.properties.

        We set option.tensor_parallel_degree & option.device_map for quantization.
        This function passes through these values to the outputted serving.properties if received from the customer.
        Otherwise, nothing is outputted for these values.
        """
        passthrough_properties = {}
        passthrough_properties[
            "option.tensor_parallel_degree"] = os.environ.get(
                "OPTION_TENSOR_PARALLEL_DEGREE") if os.environ.get(
                    "OPTION_TENSOR_PARALLEL_DEGREE"
                ) else self.customer_properties.get(
                    "option.tensor_parallel_degree")
        passthrough_properties["option.device_map"] = os.environ.get(
            "OPTION_DEVICE_MAP") if os.environ.get(
                "OPTION_DEVICE_MAP") else self.customer_properties.get(
                    "option.device_map")

        output_properties = self.properties_manager.properties
        for k, v in passthrough_properties.items():
            if v:
                logging.info(
                    f"User passed {k}={v}. Outputting in serving.properties")
                output_properties[k] = v
            else:
                output_properties.pop(k, None)
                logging.info(
                    f"User did not pass {k}. Outputted serving.properties "
                    "will not include this field.")

        self.properties_manager.properties = output_properties
        self.properties_manager.generate_properties_file()

    def neo_quantize(self):
        update_dataset_cache_location(self.HF_CACHE_LOCATION)
        self.initialize_partition_args_namespace()
        self.construct_properties_manager()
        self.parse_autofp8_config()
        self.run_quantization()
        self.write_properties()


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO,
                        force=True)

    try:
        neo_quantization_service = NeoQuantizationService()
        neo_quantization_service.neo_quantize()
    except Exception as exc:
        write_error_to_file(exc,
                            neo_quantization_service.COMPILATION_ERROR_FILE)
        raise exc


if __name__ == "__main__":
    main()
