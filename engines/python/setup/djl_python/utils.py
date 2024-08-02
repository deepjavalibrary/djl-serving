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

from djl_python import Output
from djl_python.inputs import Input


class IdCounter:

    def __init__(self):
        self.id = 0

    def get_id(self):
        return self.id

    def next_id(self):
        current_id = self.id
        self.id += 1
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
    return "best_of" in parameters.keys() and parameters.get("best_of") > 1


def is_beam_search(parameters: dict) -> bool:
    """
    Returns whether parameters indicate beam search should be applied.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "num_beams" in parameters.keys() and parameters.get("num_beams") > 1


def is_multiple_sequences(parameters: dict) -> bool:
    """
    Returns whether the parameters indicate number of output sequences to return is more than 1.
    When the user give us n, best_of is automatically applied in vllm and lmi-dist.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "n" in parameters.keys() and parameters.get("n") > 1


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
            outputs.add(Output.binary_encode(result[idx]),
                        key="data",
                        batch_index=i)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)
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
        input_data.extend(request_input.input_text)
        input_size.append(len(request_input.input_text))

        if request_input.adapters:
            adapters.extend(request_input.adapters)

        idx += 1
    adapters = adapters if adapters else None
    return input_data, input_size, parameters, adapters
