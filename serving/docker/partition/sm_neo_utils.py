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
    neo_environ = {}
    try:
        neo_environ["SM_NEO_INPUT_MODEL_DIR"] = os.environ["SM_NEO_INPUT_MODEL_DIR"]
        neo_environ["SM_NEO_COMPILED_MODEL_DIR"] = os.environ["SM_NEO_COMPILED_MODEL_DIR"]
        neo_environ["SM_NEO_COMPILATION_ERROR_FILE"] = os.environ["SM_NEO_COMPILATION_ERROR_FILE"]
        neo_environ["SM_NEO_CACHE_DIR"] = os.environ.get("SM_NEO_CACHE_DIR")
        neo_environ["SM_NEO_HF_CACHE_DIR"] = os.environ.get("SM_NEO_HF_CACHE_DIR")
        return neo_environ
    except KeyError as exc:
        raise InputConfiguration(
            f"SageMaker Neo environment variable '{exc.args[0]}' expected but not found"
            f"\nRequired env vars are: 'SM_NEO_INPUT_MODEL_DIR', 'SM_NEO_COMPILED_MODEL_DIR',"
            f" 'SM_NEO_COMPILATION_ERROR_FILE'"
        )


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


def update_dataset_cache_location(hf_cache_location):
    logging.info(
        f"Updating HuggingFace Datasets cache directory to: {hf_cache_location}"
    )
    if not os.path.isdir(hf_cache_location):
        raise InputConfiguration("Provided HF cache directory is invalid")
    os.environ['HF_DATASETS_CACHE'] = hf_cache_location
    #os.environ['HF_DATASETS_OFFLINE'] = "1"
