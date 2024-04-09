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
import argparse

from optimum.commands.export.neuronx import parse_args_neuronx

from sm_neo_utils import InputConfiguration, CompilationFatalError, write_error_to_file, get_neo_env_vars, get_neo_compiler_flags
from utils import extract_python_jar
from properties_manager import PropertiesManager
from partition import PartitionService

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'


class NeoNeuronPartitionService():

    def __init__(self):
        self.args: SimpleNamespace = SimpleNamespace()
        self.properties_manager: PropertiesManager = None
        self.properties: dict = {}
        self.compiler_flags: dict = None
        self.compiler_interface: str = None

        env = get_neo_env_vars()
        self.NEO_COMPILER_OPTIONS: Final[str] = env[0]
        self.INPUT_MODEL_DIRECTORY: Final[str] = env[1]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = env[2]
        self.COMPILATION_ERROR_FILE: Final[str] = env[3]
        self.COMPILER_CACHE_LOCATION: Final[str] = env[4]

    def update_neuron_cache_location(self):
        os.environ['NEURON_COMPILE_CACHE_URL'] = self.COMPILER_CACHE_LOCATION

    def initialize_partition_args_namespace(self):
        """
        Initialize args, a SimpleNamespace object that is used to instantiate a
        PropertiesManager for partitioning. PropertiesManager expects an
        argparse.Namespace, but we use a SimpleNamespace in its place because it
        is easier to construct.
        """
        self.args.save_mp_checkpoint_path = self.OUTPUT_MODEL_DIRECTORY
        # If skip_copy is not enabled, outputted configs are overwritten, and deployment fails.
        self.args.skip_copy = True
        # These attributes reflect the default values of the corresponding attributes in the partition argparser.
        # PropertiesManager expects these attributes to be defined.
        self.args.model_id = None
        self.args.engine = None
        self.args.tensor_parallel_degree = None

    def parse_neo_compiler_flags(self):
        if self.compiler_flags:
            if not isinstance(self.compiler_flags, dict):
                raise InputConfiguration(
                    "Invalid compiler flags. Ensure that the input is a valid JSON dictionary."
                )

            if "compile_only" in self.compiler_flags and self.compiler_flags[
                    "compile_only"].lower() == "true":
                logging.info(
                    "Warning: compile_only flag passed. SafeTensors weights or split model weights must be provided to deploy the model."
                )
                self.properties["option.partition_schema"] = "compile_only"
                del self.compiler_flags["compile_only"]

            if "compiler_interface" in self.compiler_flags:
                self.compiler_interface = self.compiler_flags[
                    "compiler_interface"]
                del self.compiler_flags["compiler_interface"]
                logging.info(
                    f"{self.compiler_interface} is set as the compiler interface by Neo CompilerOptions."
                )

    @staticmethod
    def convert_tnx_options_to_djl_options(options: dict) -> dict:
        """
        Converts Transformers-NeuronX options accepted by Neo to the equivalent option in djl-serving.
        Only options that have a different name or set of values are converted; the remaining are kept as-is.
        Supports an additional option "continuous_batching" (equivalently "batch_size_for_shared_caches") to allow
        users to enable continuous batching.

        :param options: A dictionary containing Transformers-NeuronX options as key-value pairs.
        :return: returns the modified dictionary
        """
        amp_dtype_map = {
            'f32': 'fp32',
            'f16': 'fp16',
            'bf16': 'bf16',
            's8': 'int8'
        }

        if "amp" in options:
            options["option.dtype"] = amp_dtype_map[options["amp"]]
            del options["amp"]
        if "tp_degree" in options:
            options["option.tensor_parallel_degree"] = options["tp_degree"]
            del options["tp_degree"]
        if "continuous_batching" in options or "batch_size_for_shared_caches" in options:
            max_rolling_batch_size = options[
                "continuous_batching"] if options.get(
                    "continuous_batching"
                ) else options["batch_size_for_shared_caches"]
            options["option.rolling_batch"] = "auto"
            options["option.max_rolling_batch_size"] = max_rolling_batch_size

            options.pop("continuous_batching", None)
            options.pop("batch_size_for_shared_caches", None)
            # can't set batch_size and max_rolling_batch_size at the same time
            options.pop("batch_size", None)

        return options

    def construct_properties_manager_from_tnx_options(self):
        """
        Factory method used to construct a PropertiesManager from Transformers-NeuronX Neo CompilerOptions
        """
        self.args.engine = "Python"
        # Passing a dummy location because it's expected by PropertiesManager
        self.args.properties_dir = "/dev/null"

        self.properties["model_dir"] = self.INPUT_MODEL_DIRECTORY
        self.properties["option.model_loader"] = "tnx"
        self.properties |= NeoNeuronPartitionService.convert_tnx_options_to_djl_options(
            self.compiler_flags)
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    @staticmethod
    def convert_optimum_flags_to_djl_options(
            flags: argparse.Namespace) -> dict:
        """
        This takes a namespace created by parsing Optimum CLI flags and maps the values to djlserving options.

        :param flags: a mamespace object returned by the Optimum ArgumentParser
        :return: dictionary containing the converted options
        """
        OPTIMUM_TO_DJL_MAP = {
            "task": "option.task",
            "trust-remote-code": "option.trust_remote_code",
            "auto_cast_type": "option.dtype",
            "num_cores": "option.tensor_parallel_degree",
            # rolling batch must be true for optimum
            "batch_size": "option.max_rolling_batch_size",
            "sequence_length": "option.n_positions"
        }

        props = {}

        # Iterating through the attributes of the namespace
        for flag, value in vars(flags).items():
            if flag == "task" and value == "auto":
                continue

            if flag in OPTIMUM_TO_DJL_MAP:
                props[OPTIMUM_TO_DJL_MAP[flag]] = value
                if flag == "batch_size":
                    props["option.rolling_batch"] = "auto"
            elif flag == "01":
                props["option.neuron_optimize_level"] = 1
            elif flag == "02":
                props["option.neuron_optimize_level"] = 2
            elif flag == "03":
                props["option.neuron_optimize_level"] = 3

        return props

    def construct_properties_manager_from_optimum_flags(self):
        """
        Factory method used to construct a PropertiesManager from Optimum Flags Neo CompilerOptions
        """
        self.properties["model_dir"] = self.INPUT_MODEL_DIRECTORY
        self.properties["option.model_loader"] = "optimum"
        self.args.engine = "Python"
        # Passing a dummy location because it's expected by PropertiesManager
        self.args.properties_dir = "/dev/null"

        # Construct the optimum CLI argparser
        parser = argparse.ArgumentParser()
        # This function adds the desired arguments to the parser. Not parsing yet.
        parse_args_neuronx(parser)
        # The parser requries the --model param and a trailing output parameter.
        # Passing the dummy arguments so we can still use the parser.
        optimum_flags = self.compiler_flags["flags"].split()
        optimum_flags = ["--model", "/dev/null"] + optimum_flags
        optimum_flags += ["/dev/null"]
        flags = parser.parse_args(optimum_flags)
        self.properties |= NeoNeuronPartitionService.convert_optimum_flags_to_djl_options(
            flags)

        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    def construct_properties_manager_from_serving_properties(self):
        self.args.properties_dir = self.INPUT_MODEL_DIRECTORY
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    def neo_partition(self):
        self.update_neuron_cache_location()
        self.initialize_partition_args_namespace()
        self.compiler_flags = get_neo_compiler_flags(self.NEO_COMPILER_OPTIONS)
        self.parse_neo_compiler_flags()
        if self.compiler_interface == "transformers-neuronx":
            self.construct_properties_manager_from_tnx_options()
        elif self.compiler_interface == "optimum":
            self.construct_properties_manager_from_optimum_flags()
        elif self.compiler_interface:
            raise InputConfiguration(
                f"Invalid compiler interface provided. Got: {self.compiler_interface}"
            )
        else:
            logging.info(
                "Reading compiler options from provided serving.properties file"
            )
            self.construct_properties_manager_from_serving_properties()

        logging.info(f"Model options: {self.properties_manager.properties}")

        partition_service = PartitionService(self.properties_manager)
        extract_python_jar(PYTHON_CACHE_DIR)
        try:
            partition_service.run_partition()
        except Exception as exc:
            raise CompilationFatalError(
                f"Encountered an error during Transformers-NeuronX compilation: {exc}"
            )


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO,
                        force=True)

    try:
        neo_neuron_partition_service = NeoNeuronPartitionService()
        neo_neuron_partition_service.neo_partition()
    except Exception as exc:
        write_error_to_file(
            exc, neo_neuron_partition_service.COMPILATION_ERROR_FILE)
        raise exc


if __name__ == "__main__":
    main()
