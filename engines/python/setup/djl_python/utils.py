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
from typing import Union, Callable, Any, List, Dict

from djl_python import Output
from djl_python.inputs import Input
from djl_python.encode_decode import decode
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request
from djl_python.three_p.three_p_utils import is_3p_request, parse_3p_request
from dataclasses import dataclass, field


@dataclass
class ParsedInput:
    input_data: List[str]
    input_size: List[int]
    parameters: List[dict]
    errors: dict
    batch: list
    is_client_side_batch: list = field(default_factory=lambda: [])
    adapters: list = None


@dataclass
class InputFormatConfigs:
    is_rolling_batch: bool = False
    is_adapters_supported: bool = False
    output_formatter: Union[str, Callable] = None
    tokenizer: Any = None


def parse_input_with_formatter(inputs: Input,
                               input_format_configs: InputFormatConfigs,
                               adapter_registry: dict = {}) -> ParsedInput:
    """
    Preprocessing function that extracts information from Input objects.
    :param input_format_configs: format configurations for the input.
    :param inputs :(Input) a batch of inputs, each corresponding to a new request

    :return parsed_input: object of data class that contains all parsed input details
    """

    input_data = []
    input_size = []
    parameters = []
    adapters = []
    errors = {}
    found_adapters = False
    batch = inputs.get_batches()
    # only for dynamic batch
    is_client_side_batch = [False for _ in range(len(batch))]
    for i, item in enumerate(batch):
        try:
            content_type = item.get_property("Content-Type")
            invoke_type = item.get_property("X-Amzn-SageMaker-Forwarded-Api")
            input_map = decode(item, content_type)
            _inputs, _param, is_client_side_batch[i] = _parse_inputs_params(
                input_map, item, input_format_configs, invoke_type)
            if input_format_configs.is_adapters_supported:
                adapters_per_item, found_adapter_per_item = _parse_adapters(
                    _inputs, input_map, item, adapter_registry)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"Parse input failed: {i}")
            input_size.append(0)
            errors[i] = str(e)
            continue

        input_data.extend(_inputs)
        input_size.append(len(_inputs))

        if input_format_configs.is_adapters_supported:
            adapters.extend(adapters_per_item)
            found_adapters = found_adapter_per_item or found_adapters

        for _ in range(input_size[i]):
            parameters.append(_param)

    if found_adapters and adapters is not None:
        adapter_data = [
            adapter_registry.get(adapter, None) for adapter in adapters
        ]
    else:
        adapter_data = None

    return ParsedInput(input_data=input_data,
                       input_size=input_size,
                       parameters=parameters,
                       errors=errors,
                       batch=batch,
                       is_client_side_batch=is_client_side_batch,
                       adapters=adapter_data)


def _parse_inputs_params(input_map, item, input_format_configs, invoke_type):
    if is_chat_completions_request(input_map):
        _inputs, _param = parse_chat_completions_request(
            input_map, input_format_configs.is_rolling_batch,
            input_format_configs.tokenizer)
    elif is_3p_request(invoke_type):
        _inputs, _param = parse_3p_request(
            input_map, input_format_configs.is_rolling_batch,
            input_format_configs.tokenizer, invoke_type)
    else:
        _inputs = input_map.pop("inputs", input_map)
        _param = input_map.pop("parameters", {})

    # Add some additional parameters that are necessary.
    # Per request streaming is only supported by rolling batch
    if input_format_configs.is_rolling_batch:
        _param["stream"] = input_map.pop("stream", _param.get("stream", False))

    if "cached_prompt" in input_map:
        _param["cached_prompt"] = input_map.pop("cached_prompt")
    if "seed" not in _param:
        # set server provided seed if seed is not part of request
        if item.contains_key("seed"):
            _param["seed"] = item.get_as_string(key="seed")
    if not "output_formatter" in _param:
        _param["output_formatter"] = input_format_configs.output_formatter

    if isinstance(_inputs, list):
        return _inputs, _param, True
    else:
        return [_inputs], _param, False


def _parse_adapters(_inputs, input_map, item,
                    adapter_registry) -> (List, bool):
    adapters_per_item = _fetch_adapters_from_input(input_map, item)
    found_adapter_per_item = False
    if adapters_per_item:
        _validate_adapters(adapters_per_item, adapter_registry)
        found_adapter_per_item = True
    else:
        # inference with just base model.
        adapters_per_item = [""] * len(_inputs)

    if len(_inputs) != len(adapters_per_item):
        raise ValueError(
            f"Number of adapters is not equal to the number of inputs")
    return adapters_per_item, found_adapter_per_item


def _fetch_adapters_from_input(input_map: dict, inputs: Input):
    adapters_per_item = []
    if "adapters" in input_map:
        adapters_per_item = input_map.pop("adapters", [])

    # check content, possible in workflow approach
    if inputs.contains_key("adapter"):
        adapters_per_item = inputs.get_as_string("adapter")

    # check properties, possible from header
    if "adapter" in inputs.get_properties():
        adapters_per_item = inputs.get_properties()["adapter"]

    if not isinstance(adapters_per_item, list):
        adapters_per_item = [adapters_per_item]

    return adapters_per_item


def _validate_adapters(adapters_per_item, adapter_registry):
    for adapter_name in adapters_per_item:
        if adapter_name and adapter_name not in adapter_registry:
            raise ValueError(f"Adapter {adapter_name} is not registered")


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


def is_multiple_sequences(parameters: dict) -> bool:
    """
    Returns whether the parameters indicate number of output sequences to return is more than 1.
    When the user give us n, best_of is automatically applied in vllm and lmi-dist.
    :param parameters: parameters dictionary
    :return: boolean
    """
    return "n" in parameters.keys() and parameters.get("n") > 1


def wait_till_generation_finished(parameters):
    return is_best_of(parameters) or is_multiple_sequences(parameters)


def rolling_batch_inference(parsed_input: ParsedInput, inputs: Input,
                            outputs: Output, rolling_batch):
    if inputs.get_property("reset_rollingbatch"):
        rolling_batch.reset()
    result = rolling_batch.inference(parsed_input.input_data,
                                     parsed_input.parameters,
                                     adapters=parsed_input.adapters)
    idx = 0
    for i in range(len(parsed_input.batch)):
        err = parsed_input.errors.get(i)
        if err:
            err = {"data": "", "last": True, "code": 424, "error": err}
            outputs.add(Output.binary_encode(err), key="data", batch_index=i)
            outputs.add_property(f"batch_{i}_Content-Type", "application/json")
        else:
            content_type = result[idx].pop("content_type")
            outputs.add(Output.binary_encode(result[idx]),
                        key="data",
                        batch_index=i)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)
            idx += 1
    return outputs
