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
from dataclasses import dataclass, field
from typing import List, Union, Callable, Any

from djl_python import Input
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request
from djl_python.encode_decode import decode
from djl_python.three_p.three_p_utils import is_3p_request, parse_3p_request


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
            input_size.append(0)
            err_msg = "Input Parsing failed. Ensure that the request payload is valid. "
            # str(e) for KeyError only yields the name of the key, which isn't useful as a response to the client
            if isinstance(e, KeyError):
                err_msg += f"Invalid Request Property: {e}"
            else:
                err_msg += str(e)
            errors[i] = err_msg
            logging.warning(err_msg, exc_info=True)
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
