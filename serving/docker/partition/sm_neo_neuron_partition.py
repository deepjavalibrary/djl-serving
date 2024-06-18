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
from typing import Final, Optional
import argparse
import re
import shutil

from optimum.commands.export.neuronx import parse_args_neuronx

from sm_neo_utils import (InputConfiguration, CompilationFatalError,
                          write_error_to_file, get_neo_env_vars,
                          get_neo_compiler_flags, load_jumpstart_metadata)
from utils import extract_python_jar
from properties_manager import PropertiesManager
from partition import PartitionService

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'
_neuronxcc_version: Optional[str] = None


def get_neuronxcc_version() -> str:
    """
    Gets version of NeuronX Compiler
    Copied from djl-serving/engines/python/setup/djl_python/neuron_utils/utils.py

    :return: NeuronX compiler version
    """
    global _neuronxcc_version
    if _neuronxcc_version is not None:
        return _neuronxcc_version
    try:
        import neuronxcc
    except ImportError:
        raise ModuleNotFoundError(
            "NeuronX Compiler python package is not installed.")
    _neuronxcc_version = neuronxcc.__version__
    return _neuronxcc_version


class NeoNeuronCacheManager():
    """
    This class manages the creation of alternate Neuron cache formats for Neo.
    """
    NEURONXCC_DIR = f"neuronxcc-{get_neuronxcc_version()}"

    def __init__(self, neuron_compilation_log: str, cache_dir: str):
        self.module_ids: list[
            str] = NeoNeuronCacheManager.get_neuron_cache_module_ids(
                neuron_compilation_log)
        self.cache_dir: str = cache_dir

    @staticmethod
    def get_neuron_cache_module_ids(text: str) -> list[str]:
        """
        Searches the input text for Neuron cache module IDs.
        These module IDs correspond to NEFF files in the Neuron cache needed to compile the model.

        :return: A list of unique module IDs found in the input
        """
        uniq_matches = list(set(re.findall(r"MODULE\_\w{20}\+\w{8}", text)))
        assert len(
            uniq_matches
        ) > 0, "Unable to find any module IDs in the Neuron compilation logs"
        logging.info(
            f"The Neuron cache model IDs for this model are: {uniq_matches}")
        return uniq_matches

    def copy_neuron_cache_modules(self, output_dir: str):
        """
        Copies the Neuron Cache modules for the current model to the specified output directory.
        """
        logging.info(
            f"Saving Neuron Cache NEFFs to {os.path.abspath(output_dir)}")

        for module_id in self.module_ids:
            src_path = os.path.join(self.cache_dir, self.NEURONXCC_DIR,
                                    module_id)
            dst_path = os.path.join(output_dir, module_id)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    def create_jumpstart_neuron_cache_in_output_dir(self, output_dir: str):
        """
        Saves the Neuron cache in JumpStart format in the given directory.
        The format in this case is:
        <output dir>/PRE_COMPILED_NEURON_GRAPH_INFER/<Neuron Persistent Cache format>
        For example:
        <output dir>/PRE_COMPILED_NEURON_GRAPH_INFER/neuronxcc-2.13.68.0+6dfecc895/<Module folders>

        :param output_dir: the path to saved to. Intended to be the partitioning output directory.
        """
        output_dir = os.path.join(output_dir,
                                  "PRE_COMPILED_NEURON_GRAPH_INFER",
                                  self.NEURONXCC_DIR)

        logging.info(
            "Saving Neuron Cache NEFFs for JumpStart in partition output directory"
        )
        self.copy_neuron_cache_modules(output_dir)

    def create_jumpstart_neuron_cache_in_cache_dir(self,
                                                   jumpstart_metadata: dict):
        """
        Saves the Neuron cache using the passed metadata in the Neuron cache directory with the
        following format:
        /<cache dir>/JUMPSTART_COMPILED_GRAPHS/neuronxcc-<compiler version>/<JumpStart model id>/...
        /<JumpStart model scope>/PRE_COMPILED_NEURON_GRAPH_INFER/<Neuron Persistent Cache format>

        For example:
        /<cache dir>/JUMPSTART_COMPILED_GRAPHS/neuronxcc-2.13.68.0+6dfecc895/...
        /<JumpStart model id>/inference/PRE_COMPILED_NEURON_GRAPH_INFER/...
        /neuronxcc-2.13.68.0+6dfecc895/<Module folders>
        """
        try:
            jumpstart_model_id = jumpstart_metadata["model_info"]["model_id"]
            jumpstart_model_scope = jumpstart_metadata["script_info"]["Scope"]
        except KeyError as key:
            logging.warning(
                f"Missing field {key} from JumpStart metadata. "
                "JumpStart cache will not be constructed in Neuron Cache directory"
            )
            return

        output_dir = os.path.join(self.cache_dir, "JUMPSTART_COMPILED_GRAPHS",
                                  self.NEURONXCC_DIR, jumpstart_model_id,
                                  jumpstart_model_scope,
                                  "PRE_COMPILED_NEURON_GRAPH_INFER",
                                  self.NEURONXCC_DIR)

        logging.info(
            "Saving Neuron Cache NEFFs for JumpStart in Neuron cache directory"
        )
        self.copy_neuron_cache_modules(output_dir)


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
        # Specific to Neuron compilation
        self.CACHE_JUMPSTART_FORMAT: Final[str] = os.environ.get(
            "SM_CACHE_JUMPSTART_FORMAT", "false")

        self.jumpstart_metadata: dict = load_jumpstart_metadata(
            self.INPUT_MODEL_DIRECTORY)

    def update_neuron_cache_location(self):
        logging.info(
            f"Updating Neuron Persistent Cache directory to: {self.COMPILER_CACHE_LOCATION}"
        )
        os.environ['NEURON_COMPILE_CACHE_URL'] = self.COMPILER_CACHE_LOCATION

    def initialize_partition_args_namespace(self):
        """
        Initialize args, a SimpleNamespace object that is used to instantiate a
        PropertiesManager for partitioning. PropertiesManager expects an
        argparse.Namespace, but we use a SimpleNamespace in its place because it
        is easier to construct.
        """
        self.args.save_mp_checkpoint_path = self.OUTPUT_MODEL_DIRECTORY
        self.args.engine = "Python"
        # If skip_copy is not enabled, outputted configs are overwritten, and deployment fails.
        self.args.skip_copy = True
        # These attributes reflect the default values of the corresponding attributes
        # in the partition argparser. PropertiesManager expects these attributes to be defined.
        self.args.model_id = None
        self.args.tensor_parallel_degree = None
        self.args.quantize = None

    def parse_neo_compiler_flags(self):
        """
        Parses the compiler_flags field of Neo CompilerOptions.
        """
        if self.compiler_flags:
            logging.debug(f"Compiler Flags: {self.compiler_flags}")
            if not isinstance(self.compiler_flags, dict):
                raise InputConfiguration(
                    "Invalid compiler flags. Ensure that the input is a valid JSON dictionary."
                )

            if "compile_only" in self.compiler_flags and self.compiler_flags[
                    "compile_only"].lower() == "true":
                logging.info(
                    "Warning: compile_only flag passed. SafeTensors weights or split model "
                    "weights must be provided to deploy the model.")
                self.properties["option.partition_schema"] = "compile_only"
                del self.compiler_flags["compile_only"]

            if "compiler_interface" in self.compiler_flags:
                self.compiler_interface = self.compiler_flags[
                    "compiler_interface"]
                del self.compiler_flags["compiler_interface"]
                logging.info(
                    f"{self.compiler_interface} is set as the compiler interface by "
                    "Neo CompilerOptions.")

    @staticmethod
    def convert_tnx_options_to_djl_options(options: dict) -> dict:
        """
        Converts Transformers-NeuronX options accepted by Neo to the equivalent option
        in djl-serving. Only options that have a different name or set of values are converted;
        the remaining are kept as-is. Supports an additional option "continuous_batching"
        (equivalently "batch_size_for_shared_caches") to allow users to enable continuous batching.

        :param options: A dictionary containing Transformers-NeuronX options as key-value pairs.
        :return: returns the modified dictionary
        """
        amp_dtype_map = {
            'f32': 'fp32',
            'f16': 'fp16',
            'bf16': 'bf16',
            's8': 'int8'
        }

        logging.debug(f"tnx options dict: {options}")
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
        Factory method used to construct a PropertiesManager from Transformers-NeuronX
        Neo CompilerOptions
        """
        self.args.engine = "Python"
        # Passing a dummy location because it's expected by PropertiesManager
        self.args.properties_dir = "/dev/null"

        self.properties["model_dir"] = self.INPUT_MODEL_DIRECTORY
        self.properties["option.model_loader"] = "tnx"
        self.properties |= NeoNeuronPartitionService.convert_tnx_options_to_djl_options(
            self.compiler_flags)
        logging.debug("Constructing PropertiesManager from TNX "
                      f"options\nargs:{self.args}\nprops:{self.properties}")
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    @staticmethod
    def convert_optimum_flags_to_djl_options(
            flags: argparse.Namespace) -> dict:
        """
        This takes a namespace created by parsing Optimum CLI flags and maps the values to
        djl-serving options.

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

        logging.debug(f"optimum flags: {flags}")

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

        logging.debug("Constructing PropertiesManager from optimum "
                      f"options\nargs:{self.args}\nprops:{self.properties}")
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    def construct_properties_manager_from_serving_properties(self):
        """
        Factory method used to construct a PropertiesManager from serving.properties
        """
        self.args.properties_dir = self.INPUT_MODEL_DIRECTORY
        logging.debug(
            "Constructing PropertiesManager from "
            f"serving.properties\nargs:{self.args}\nprops:{self.properties}")
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)

    def run_partition(self) -> str:
        """
        :return: the output of the partition command captured from stdout
        """
        partition_service = PartitionService(self.properties_manager)
        extract_python_jar(PYTHON_CACHE_DIR)
        try:
            return partition_service.run_partition()
        except Exception as exc:
            raise CompilationFatalError(
                f"Encountered an error during Transformers-NeuronX compilation: {exc}"
            )

    def neo_partition(self):
        self.update_neuron_cache_location()
        self.initialize_partition_args_namespace()
        self.compiler_flags = get_neo_compiler_flags(self.NEO_COMPILER_OPTIONS)
        self.parse_neo_compiler_flags()
        if self.compiler_interface == "transformers-neuronx":
            logging.info(
                "Reading Neo CompilerOptions in transformers-neuronx format. "
                "serving.properties will be ignored")
            self.construct_properties_manager_from_tnx_options()
            logging.info("Reading Neo CompilerOptions in optimum format. "
                         "serving.properties will be ignored")
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
        partition_output = self.run_partition()

        if self.CACHE_JUMPSTART_FORMAT.lower() == 'true':
            logging.info("JumpStart cache environment variable is set")
            if self.jumpstart_metadata:
                logging.info(
                    "JumpStart metadata found. Outputting NEFFs in JumpStart format"
                )
                cache_manager = NeoNeuronCacheManager(
                    partition_output, self.COMPILER_CACHE_LOCATION)
                cache_manager.create_jumpstart_neuron_cache_in_output_dir(
                    self.OUTPUT_MODEL_DIRECTORY)
                cache_manager.create_jumpstart_neuron_cache_in_cache_dir(
                    self.jumpstart_metadata)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO,
                        force=True)

    try:
        neo_neuron_partition_service = NeoNeuronPartitionService()
        neo_neuron_partition_service.neo_partition()
    except Exception as exc:
        write_error_to_file(exc,
                            os.environ.get("SM_NEO_COMPILATION_ERROR_FILE"))
        raise exc


if __name__ == "__main__":
    main()
