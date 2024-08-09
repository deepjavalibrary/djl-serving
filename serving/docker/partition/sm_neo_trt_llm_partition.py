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

import logging
import os
import sys
from typing import Final

from utils import (update_kwargs_with_env_vars, load_properties,
                   remove_option_from_properties)
from sm_neo_utils import (CompilationFatalError, write_error_to_file,
                          update_dataset_cache_location, get_neo_env_vars)
from tensorrt_llm_toolkit import create_model_repo


class NeoTRTLLMPartitionService():

    def __init__(self):
        self.properties: dict = {}

        neo_environ = get_neo_env_vars()
        self.INPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_INPUT_MODEL_DIR"]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_COMPILED_MODEL_DIR"]
        self.COMPILATION_ERROR_FILE: Final[str] = neo_environ[
            "SM_NEO_COMPILATION_ERROR_FILE"]
        self.COMPILER_CACHE_LOCATION: Final[str] = neo_environ[
            "SM_NEO_CACHE_DIR"]
        self.HF_CACHE_LOCATION: Final[str] = neo_environ["SM_NEO_HF_CACHE_DIR"]

    def run_partition(self):
        kwargs = remove_option_from_properties(self.properties)
        kwargs["trt_llm_model_repo"] = self.OUTPUT_MODEL_DIRECTORY
        kwargs["neo_cache_dir"] = self.COMPILER_CACHE_LOCATION
        os.environ['TRTLLM_TOOLKIT_SKIP_DOWNLOAD_DIR_CLEANUP'] = 'true'
        os.environ['TRTLLM_TOOLKIT_SKIP_CHECKPOINT_DIR_CLEANUP'] = 'true'
        try:
            create_model_repo(self.INPUT_MODEL_DIRECTORY, **kwargs)
        except Exception as exc:
            raise CompilationFatalError(
                f"Encountered an error during TRT-LLM compilation: {exc}")

    def get_properties(self):
        """Get properties from serving.properties and/or environment variables."""
        self.properties = update_kwargs_with_env_vars({})
        self.properties.update(load_properties(self.INPUT_MODEL_DIRECTORY))

    def generate_properties_file(self):
        """Generate serving.properties file in output repo, so compiled artifacts can be deployed."""
        with open(
                os.path.join(self.OUTPUT_MODEL_DIRECTORY,
                             "serving.properties"), "w") as f:
            f.write("engine=MPI\n")
            for key, value in self.properties.items():
                if key != "option.model_id" and key != "option.model_dir":
                    f.write(f"{key}={value}\n")

    def neo_partition(self):
        update_dataset_cache_location(self.HF_CACHE_LOCATION)
        self.get_properties()
        self.run_partition()
        self.generate_properties_file()


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO,
                        force=True)

    try:
        neo_trtllm_partition_service = NeoTRTLLMPartitionService()
        neo_trtllm_partition_service.neo_partition()
    except Exception as exc:
        write_error_to_file(
            exc, neo_trtllm_partition_service.COMPILATION_ERROR_FILE)
        raise exc


if __name__ == "__main__":
    main()
