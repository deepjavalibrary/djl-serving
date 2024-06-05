"""
This module contains functions and exceptions used by Neo partition scripts
"""
import json
import logging
import os
import traceback


class InputConfiguration(Exception):
    """Raise when SageMaker Neo interface expectation is not met"""


class CompilationFatalError(Exception):
    """Raise for errors encountered during the TensorRT-LLM build process"""


def write_error_to_file(error_message, error_file):
    """
    Write error messages to error file
    """
    try:
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump({"error": repr(error_message)}, f)
    except:
        tb_exc = traceback.format_exc()
        logging.error(f"Failed to write error file: {tb_exc}")


def get_neo_env_vars():
    """
    Get environment variables required by the SageMaker Neo interface
    """
    try:
        compiler_options = os.environ.get("COMPILER_OPTIONS")
        input_model_directory = os.environ["SM_NEO_INPUT_MODEL_DIR"]
        compiled_model_directory = os.environ["SM_NEO_COMPILED_MODEL_DIR"]
        compilation_error_file = os.environ["SM_NEO_COMPILATION_ERROR_FILE"]
        neo_cache_dir = os.environ["SM_NEO_CACHE_DIR"]
        return (compiler_options, input_model_directory,
                compiled_model_directory, compilation_error_file,
                neo_cache_dir)
    except KeyError as exc:
        raise InputConfiguration(
            f"SageMaker Neo environment variable '{exc.args[0]}' expected but not found"
            f" \nRequired env vars are: 'COMPILER_OPTIONS', 'SM_NEO_INPUT_MODEL_DIR',"
            f" 'SM_NEO_COMPILED_MODEL_DIR', 'SM_NEO_COMPILATION_ERROR_FILE', 'SM_NEO_CACHE_DIR'"
        )


def get_neo_compiler_flags(compiler_options):
    """
    Get SageMaker Neo compiler_flags from the CompilerOptions field
    """
    try:
        # CompilerOptions JSON will always be present, but compiler_flags key is optional
        compiler_options = json.loads(compiler_options)
        logging.info(f"Parsing CompilerOptions: {compiler_options}")
        if not isinstance(compiler_options, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return compiler_options.get("compiler_flags")
    except Exception as exc:
        raise InputConfiguration(
            f"Failed to parse SageMaker Neo CompilerOptions: {exc}")


def load_jumpstart_metadata(path: str) -> dict:
    """
    Loads the JumpStart metadata files __model_info__.json, __script_info__.json files to a
    dictionary if they exist.
    """
    js_metadata = {}
    model_info_path = os.path.join(path, "__model_info__.json")
    script_info_path = os.path.join(path, "__script_info__.json")

    if os.path.exists(model_info_path):
        logging.info("JumpStart __model_info__.json found")
        with open(model_info_path) as file:
            js_metadata["model_info"] = json.load(file)

    if os.path.exists(script_info_path):
        logging.info("JumpStart __script_info__.json found")
        with open(script_info_path) as file:
            js_metadata["script_info"] = json.load(file)

    return js_metadata
