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
import sys

from utils import load_properties
from sm_neo_utils import InputConfiguration, CompilationFatalError, write_error_to_file, get_neo_env_vars, get_neo_compiler_flags
from tensorrt_llm_toolkit import create_model_repo

# TODO: Merge the functionality of this file into trt_llm_partition.py
# so all TRT-LLM partitioning is unified

DJL_SERVING_OPTION_PREFIX = "option."


def verify_neo_compiler_flags(compiler_flags):
    """
    Verify that provided compiler_flags is a valid configuration
    """
    convert_checkpoint_flags = compiler_flags.get("convert_checkpoint_flags")
    quantize_flags = compiler_flags.get("quantize_flags")
    trtllm_build_flags = compiler_flags.get("trtllm_build_flags")

    if trtllm_build_flags is None:
        raise InputConfiguration(
            "`compiler_flags` were found, but required sub-field `trtllm_build_flags` was not defined."
            " See SageMaker Neo documentation for more info:"
            " https://docs.aws.amazon.com/sagemaker/latest/dg/neo-troubleshooting.html"
        )
    if convert_checkpoint_flags is None and quantize_flags is None:
        raise InputConfiguration(
            "`compiler_flags` were found, but neither sub-fields `convert_checkpoint_flags` "
            " or `quantize_flags` were defined, at least one of which must be provided."
            " See SageMaker Neo documentation for more info:"
            " https://docs.aws.amazon.com/sagemaker/latest/dg/neo-troubleshooting.html"
        )
    if convert_checkpoint_flags is not None and quantize_flags is not None:
        logging.warning(
            "Both `convert_checkpoint_flags` and `quantize_flags` were provided –"
            " `convert_checkpoint_flags` will be used.")


def main():
    """
    Convert from SageMaker Neo interface to DJL-Serving format for TRT-LLM compilation
    """
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)

    compilation_error_file = None
    try:
        (compiler_options, input_model_directory, compiled_model_directory,
         compilation_error_file, neo_cache_dir, neo_hf_cache_dir,
         neo_deployment_instance_type) = get_neo_env_vars()

        # Neo requires that serving.properties is in same dir as model files
        serving_properties = load_properties(input_model_directory)
        compiler_flags = get_neo_compiler_flags(compiler_options)
        kwargs = {}

        if compiler_flags is not None:
            # If set, prefer Neo CompilerOptions flags over LMI serving.properties
            logging.info(
                f"Using CompilerOptions from SageMaker Neo. If a `serving.properties`"
                " file was provided, it will be ignored for compilation.")
            verify_neo_compiler_flags(compiler_flags)
            kwargs = compiler_flags
        elif len(serving_properties) > 0:
            # Else, if present, use LMI serving.properties options
            logging.info(
                f"Using compiler options from serving.properties file")

            for key, value in serving_properties.items():
                if key.startswith(DJL_SERVING_OPTION_PREFIX):
                    kwargs[key[len(DJL_SERVING_OPTION_PREFIX):]] = value
                else:
                    kwargs[key] = value
        else:
            raise InputConfiguration(
                "Neither compiler_flags nor serving.properties found. Please either:"
                " \nA) specify `compiler_flags` in the CompilerOptions field of SageMaker Neo or CreateCompilationJob API, or"
                " \nB) include a `serving.properties` file along with your model files."
                " \nFor info on valid `compiler_flags` fields and values for TensorRT-LLM, see SageMaker Neo documentation:"
                " https://docs.aws.amazon.com/sagemaker/latest/dg/neo-troubleshooting.html"
                " \nFor `serving.properties` configuration, see"
                " https://docs.djl.ai/docs/serving/serving/docs/lmi/user_guides/trt_llm_user_guide.html"
                " Note that SageMaker Neo requires that the `serving.properties` file is placed in the"
                " same directory as the model files, i.e. on the same level as `config.json` and checkpoints."
            )

        try:
            kwargs["trt_llm_model_repo"] = compiled_model_directory
            kwargs["neo_cache_dir"] = neo_cache_dir
            create_model_repo(input_model_directory, **kwargs)
        except Exception as exc:
            raise CompilationFatalError(
                f"Encountered an error during TRT-LLM compilation: {exc}")

    except Exception as exc:
        write_error_to_file(exc, compilation_error_file)
        raise exc


if __name__ == "__main__":
    main()
