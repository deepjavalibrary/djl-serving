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
import re
import shutil

from sm_neo_utils import (InputConfiguration, CompilationFatalError,
                          write_error_to_file, get_neo_env_vars,
                          load_jumpstart_metadata)
from utils import extract_python_jar, load_properties
from properties_manager import PropertiesManager
from partition import PartitionService

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'
_neuronxcc_version: Optional[str] = None
NEO_OPTIMIZED_MODEL_DIR = 'optimized_model'


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
            jumpstart_model_id = jumpstart_metadata["model_info"]["ModelId"]
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

        neo_environ = get_neo_env_vars()
        self.INPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_INPUT_MODEL_DIR"]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_COMPILED_MODEL_DIR"]
        self.COMPILATION_ERROR_FILE: Final[str] = neo_environ[
            "SM_NEO_COMPILATION_ERROR_FILE"]
        self.COMPILER_CACHE_LOCATION: Final[str] = neo_environ[
            "SM_NEO_CACHE_DIR"]
        # Specific to Neuron compilation
        self.CACHE_JUMPSTART_FORMAT: Final[str] = os.environ.get(
            "SM_CACHE_JUMPSTART_FORMAT", "false")

        self.jumpstart_metadata: dict = load_jumpstart_metadata(
            self.INPUT_MODEL_DIRECTORY)

    def update_neuron_cache_location(self):
        logging.info(
            f"Updating Neuron Persistent Cache directory to: {self.COMPILER_CACHE_LOCATION}"
        )
        if not os.path.isdir(self.COMPILER_CACHE_LOCATION):
            raise InputConfiguration("Provided Neo cache directory is invalid")
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
        self.args.pipeline_parallel_degree = None
        self.args.quantize = None

    def construct_properties_manager_from_serving_properties(self):
        """
        Factory method used to construct a PropertiesManager from serving.properties
        """
        self.args.properties_dir = self.INPUT_MODEL_DIRECTORY
        self.properties[
            "option.entryPoint"] = "djl_python.transformers_neuronx"
        logging.debug(
            "Constructing PropertiesManager from "
            f"serving.properties\nargs:{self.args}\nprops:{self.properties}")
        self.properties_manager = PropertiesManager(
            self.args, addl_properties=self.properties)
        if not self.properties_manager.properties.get(
                "option.tensor_parallel_degree"):
            raise InputConfiguration(
                "Tensor parallel degree not specified. This is required for Neuron compilation"
            )

    def run_partition(self) -> str:
        """
        :return: the output of the partition command captured from stdout
        """
        partition_service = PartitionService(self.properties_manager)
        extract_python_jar(PYTHON_CACHE_DIR)
        try:
            return partition_service.run_partition()
        except Exception as exc:
            raise CompilationFatalError(str(exc))

    def write_properties(self) -> str:
        """
        Updates outputted serving.properties.

        engine=Python & option.entryPoint=djl_python.transformers_neuronx are hard-coded for Neo partitioning.
        This function outputs the customer inputs for these fields.
        """
        customer_properties = load_properties(self.INPUT_MODEL_DIRECTORY)
        passthrough_properties = {}
        passthrough_properties["engine"] = customer_properties.get('engine')
        passthrough_properties["option.entryPoint"] = os.environ.get(
            "OPTION_ENTRYPOINT") if os.environ.get(
                "OPTION_ENTRYPOINT") else customer_properties.get(
                    "option.entryPoint")

        output_properties = self.properties_manager.properties
        output_passthrough_properties = {}
        for k, v in passthrough_properties.items():
            output_properties.pop(k, None)
            if v:
                logging.info(
                    f"User passed {k}={v}. Outputting in serving.properties")
                output_passthrough_properties[k] = v

        # Write out properties without pass-through properties
        self.properties_manager.properties = output_properties
        self.properties_manager.generate_properties_file()

        output_passthrough_properties[
            "option.model_id"] = f"./{NEO_OPTIMIZED_MODEL_DIR}"
        # Write out pass-through properties
        properties_file = os.path.join(self.OUTPUT_MODEL_DIRECTORY,
                                       'serving.properties')
        with open(properties_file, "a") as f:
            for k, v in output_passthrough_properties.items():
                f.write(f"{k}={v}\n")

    def copy_input_files_to_output(self):
        """
        Copies input model files to output directory. Compilation outputs are
        saved in a subdirectory of the output directory, and the generated
        serving.properties sets the model location to the subdirectory.
        This is done so that that custom entrypoints or requirements.txt files are preserved.

        TODO: Avoid making redundant copies of model weights.
        """
        # move outputted files to subdirectory
        optimized_model_dir = os.path.abspath(
            os.path.join(self.OUTPUT_MODEL_DIRECTORY, NEO_OPTIMIZED_MODEL_DIR))
        os.mkdir(optimized_model_dir)
        with os.scandir(self.OUTPUT_MODEL_DIRECTORY) as it:
            for entry in it:
                if os.path.abspath(entry.path) != optimized_model_dir:
                    shutil.move(entry.path, optimized_model_dir)

        shutil.copytree(self.INPUT_MODEL_DIRECTORY,
                        self.OUTPUT_MODEL_DIRECTORY,
                        dirs_exist_ok=True)
        self.write_properties()

    def neo_partition(self):
        self.update_neuron_cache_location()
        self.initialize_partition_args_namespace()
        logging.info(
            "Reading compiler options from provided serving.properties file")
        self.construct_properties_manager_from_serving_properties()
        logging.info(f"Model options: {self.properties_manager.properties}")
        partition_output = self.run_partition()

        self.copy_input_files_to_output()

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
