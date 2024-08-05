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

from sm_neo_utils import (CompilationFatalError, InputConfiguration,
                          write_error_to_file, get_neo_env_vars)
from utils import (extract_python_jar, load_properties,
                   update_dataset_cache_location)
from properties_manager import PropertiesManager
from partition import PartitionService

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'
AUTOFP8_CONFIG_ENVVAR = 'AUTOFP8_CONFIG'


class NeoQuantizationService():

    def __init__(self):
        self.args: SimpleNamespace = SimpleNamespace()
        self.properties_manager: PropertiesManager = None
        self.compiler_flags: dict = None

        env = get_neo_env_vars()
        self.INPUT_MODEL_DIRECTORY: Final[str] = env[1]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = env[2]
        self.COMPILATION_ERROR_FILE: Final[str] = env[3]
        self.HF_CACHE_LOCATION: Final[str] = env[5]

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
        self.args.pipeline_parallel_degree = None
        self.args.model_id = None
        self.args.quantize = None
        self.args.skip_copy = None
        self.args.engine = None

    def construct_properties_manager(self):
        """
        Factory method used to construct a QuantizationPropertiesManager from
        given serving.properties
        """
        # Default to awq quantization
        if not os.environ.get('OPTION_QUANTIZE'):
            os.environ['OPTION_QUANTIZE'] = 'awq'
        logging.debug("Constructing PropertiesManager from "
                      f"serving.properties\nargs:{self.args}\n")
        self.properties_manager = PropertiesManager(self.args)

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
            raise CompilationFatalError(
                f"Encountered an error during quantization: {exc}")

    def write_properties(self):
        """
        Updates outputted serving.properties.
        If a user passes in tensor_parallel_degree, it is passed through to the output.
        Otherwise, tensor_parallel_degree is not outputted so that it can be defined
        during serving.
        """
        customer_properties = load_properties(self.INPUT_MODEL_DIRECTORY)
        user_tensor_parallel_degree = customer_properties.get(
            "option.tensor_parallel_degree")
        if os.environ.get("OPTION_TENSOR_PARALLEL_DEGREE"):
            user_tensor_parallel_degree = os.environ.get(
                "OPTION_TENSOR_PARALLEL_DEGREE")

        output_properties = self.properties_manager.properties
        if user_tensor_parallel_degree:
            logging.info(
                f"User passed tensor_parallel_degree={user_tensor_parallel_degree}"
            )
            output_properties[
                "option.tensor_parallel_degree"] = user_tensor_parallel_degree
        else:
            logging.info(
                "User did not pass tensor_parallel_degree. Outputted serving.properties "
                "will not include this field.")
            del output_properties["option.tensor_parallel_degree"]

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
