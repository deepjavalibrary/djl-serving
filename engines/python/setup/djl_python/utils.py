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
import glob
import logging
import os
import inspect
import importlib.util
from typing import Optional, List, Callable

from djl_python import Output
from djl_python.inputs import Input
from djl_python.service_loader import load_model_service, has_function_in_module

# SageMaker function signatures for validation
SAGEMAKER_SIGNATURES = {
    'model_fn': [['model_dir']],
    'input_fn': [['request_body', 'content_type'],
                 ['request_body', 'request_content_type']],
    'predict_fn': [['input_data', 'model']],
    'output_fn': [['prediction', 'accept'],
                  ['prediction', 'response_content_type']]
}


class IdCounter:
    MAX_ID = 999999

    def __init__(self):
        self.id = 0

    def get_id(self):
        return self.id

    def next_id(self):
        current_id = self.id
        self.id = (self.id + 1) % self.MAX_ID
        return current_id

    def reset(self):
        self.id = 0


def profile_objects(func):
    """
    Profile on system for object leakage
    """
    import os
    do_profiling = os.environ.get("DJL_PYTHON_PROFILING",
                                  "false").lower() == "true"
    if do_profiling:
        import objgraph
    top_objects = int(os.environ.get("DJL_PYTHON_PROFILING_TOP_OBJ", "50"))

    def apply_profiling(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if do_profiling:
            logging.info(f"Function: {func.__name__} in {func.__module__}")
            # getting top 100 objects that need tracking
            mem_tracker = objgraph.growth(limit=top_objects)
            if mem_tracker:
                final_outcome = []
                width = max(len(name) for name, _, _ in mem_tracker)
                for name, count, delta in mem_tracker:
                    final_outcome.append('%-*s%9d %+9d\n' %
                                         (width, name, count, delta))
                logging.info("".join(final_outcome))
        return result

    return apply_profiling


def is_best_of(parameters: dict) -> bool:
    """
    Returns whether parameters indicate best_of should be applied.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "best_of" in parameters and parameters.get("best_of") > 1


def is_beam_search(parameters: dict) -> bool:
    """
    Returns whether parameters indicate beam search should be applied.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "num_beams" in parameters and parameters.get("num_beams") > 1


def is_multiple_sequences(parameters: dict) -> bool:
    """
    Returns whether the parameters indicate number of output sequences to return is more than 1.
    When the user give us n, best_of is automatically applied in vllm.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "n" in parameters and parameters.get("n") > 1


def is_streaming(parameters: dict) -> bool:
    """
    Returns whether token streaming is enabled for the request
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "stream" in parameters and parameters.get("stream")


def wait_till_generation_finished(parameters):
    return is_best_of(parameters) or is_multiple_sequences(
        parameters) or is_beam_search(parameters)


def rolling_batch_inference(parsed_input, inputs: Input, outputs: Output,
                            rolling_batch):
    if inputs.get_property("reset_rollingbatch"):
        rolling_batch.reset()
    result = rolling_batch.inference(parsed_input.requests)
    idx = 0
    for i in range(len(parsed_input.batch)):
        err = parsed_input.errors.get(i)
        if err:
            err = {"data": "", "last": True, "code": 424, "error": err}
            outputs.add(Output.binary_encode(err), key="data", batch_index=i)
            outputs.add_property(f"batch_{i}_Content-Type", "application/json")
        else:
            content_type = result[idx].get("content_type")
            client_request_id = result[idx].get("request_id")
            outputs.add(Output.binary_encode(result[idx]),
                        key="data",
                        batch_index=i)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)
            outputs.add_property(f"batch_{i}_requestId", client_request_id)
            idx += 1
    return outputs


def get_input_details(requests, errors, batch):
    # Dynamic batching
    input_data = []
    input_size = []
    adapters = []
    idx = 0
    parameters = requests[0].request_input.server_parameters

    for i in range(len(batch)):
        if i in errors:
            input_size.append(0)
            continue
        request = requests[idx]
        request_input = request.request_input
        if request_input.server_parameters != parameters:
            raise ValueError(
                "In order to enable dynamic batching, all input batches must have the same parameters"
            )

        if not isinstance(request_input.input_text, list):
            request_input.input_text = [request_input.input_text]

        input_data.extend(request_input.input_text)
        input_size.append(len(request_input.input_text))

        if request_input.adapters:
            adapters.extend(request_input.adapters)

        idx += 1
    adapters = adapters if adapters else None
    return input_data, input_size, parameters, adapters


def find_model_file(model_dir: str, extensions: List[str]) -> Optional[str]:
    """Find model file with given extensions in model directory
    
    Args:
        model_dir: Directory to search for model files
        extensions: List of file extensions to search for (without dots)
        
    Returns:
        Path to matching model file, or None if not found
    """
    all_matches = []
    for ext in extensions:
        pattern = os.path.join(model_dir, f"*.{ext}")
        matches = glob.glob(pattern)
        all_matches.extend(matches)

    if len(all_matches) > 1:
        raise ValueError(
            f"Multiple model files found in {model_dir}: {all_matches}. Only one model file is supported per directory."
        )

    return all_matches[0] if all_matches else None


def _validate_sagemaker_function(module, func_name: str,
                                 expected_params) -> Optional[Callable]:
    """
    Validate that function exists and has correct signature
    Returns the function if valid, None otherwise
    """
    if not hasattr(module, func_name):
        return None

    func = getattr(module, func_name)
    if not callable(func):
        return None

    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Handle multiple signature options
        for signature_option in expected_params:
            if param_names == signature_option:
                return func
    except (ValueError, TypeError):
        # Handle cases where signature inspection fails
        pass

    return None


def get_sagemaker_function(model_dir: str,
                           func_name: str) -> Optional[Callable]:
    """
    Load and validate SageMaker-style formatter function from model.py
    
    :param model_dir: model directory containing model.py
    :param func_name: SageMaker function name (model_fn, input_fn, predict_fn, output_fn)
    :return: Validated function or None if not found/invalid
    """

    if func_name not in SAGEMAKER_SIGNATURES:
        return None

    try:
        service = load_model_service(model_dir, "model.py", -1)
        if has_function_in_module(service.module, func_name):
            func = getattr(service.module, func_name)
            # Optional: validate signature
            expected_params = SAGEMAKER_SIGNATURES[func_name]
            if _validate_sagemaker_function(service.module, func_name,
                                            expected_params):
                return func

    except Exception as e:
        logging.debug(f"Failed to load {func_name} from model.py: {e}")
        return None
