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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from djl_python import Input
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request
from djl_python.encode_decode import decode
from djl_python.properties_manager.properties import is_rolling_batch_enabled
from djl_python.request import Request
from djl_python.request_io import TextInput, RequestInput
from djl_python.three_p.three_p_utils import parse_3p_request


def input_formatter(function):
    """
    Decorator for input_formatter. User just need to annotate @input_formatter for their custom defined function.
    :param function:  Decorator takes in the function and adds an attribute.
    :return:
    """
    # adding an attribute to the function, which is used to find the decorated function.
    function.is_input_formatter = True
    return function


@dataclass
class ParsedInput:
    errors: dict = field(default_factory=lambda: {})
    requests: List[Request] = field(default_factory=lambda: [])
    batch: List = field(default_factory=lambda: [])


def get_batch_start_id(batch, **kwargs):
    if kwargs.get("is_rolling_batch"):
        # for rolling batch, we only need to parse the new requests, as the active requests kept in cache.
        rolling_batch = kwargs.get("rolling_batch")
        active_requests_len = len(rolling_batch.active_requests)
        batch_size = len(batch)
        if batch_size > active_requests_len:
            # if batch_size > active_requests_len, then new requests are received
            return active_requests_len
        else:
            # no new requests are received, so sending batch_size, nothing will be parsed.
            return batch_size
    else:
        # for non-rolling batch, python process only receives new requests.
        return 0


def parse_input_with_formatter(inputs: Input, **kwargs) -> ParsedInput:
    """
    Preprocessing function that extracts information from Input objects.
    :param inputs :(Input) a batch of inputs, each corresponding to a new request

    :return parsed_input: object of data class that contains all parsed input details
    """

    errors = {}
    requests = []
    batch = inputs.get_batches()
    configs = kwargs.get("configs")
    kwargs["is_rolling_batch"] = is_rolling_batch_enabled(
        configs.rolling_batch)
    req_id_counter = get_req_id_counter(kwargs)
    start_batch_id = get_batch_start_id(batch, **kwargs)
    input_formatter_function = configs.input_formatter if configs.input_formatter else format_input
    for i in range(start_batch_id, len(batch)):
        input_item = batch[i]
        try:
            # input formatter can be user written as well. We look for model.py and search for the decorator.
            request_input = input_formatter_function(input_item, **kwargs)

            # populate additional information in request_input
            request_id = req_id_counter.next_id() if req_id_counter else i
            request_input.request_id = request_id
            request_input.tokenizer = kwargs.get("tokenizer")
            request_input.tgi_compat = configs.tgi_compat

            # We add server maintained parameters
            add_server_maintained_params(request_input, input_item, **kwargs)
            request = Request(request_input=request_input)
            requests.append(request)
        except Exception as e:  # pylint: disable=broad-except
            err_msg = "Input Parsing failed. Ensure that the request payload is valid. "
            # str(e) for KeyError only yields the name of the key, which isn't useful as a response to the client
            if isinstance(e, KeyError):
                err_msg += f"Invalid Request Property: {e}"
            else:
                err_msg += str(e)
            errors[i] = err_msg
            logging.warning(err_msg, exc_info=True)
            continue

    return ParsedInput(errors=errors, requests=requests, batch=batch)


def get_req_id_counter(kwargs):
    req_id_counter = None
    if kwargs.get("is_rolling_batch"):
        req_id_counter = kwargs.get("rolling_batch").req_id_counter
    return req_id_counter


def format_input(input_item: Input, **kwargs) -> RequestInput:
    # TODO: Decide whether it is a text input based on content-type
    request_input = TextInput()
    content_type = input_item.get_property("Content-Type")
    input_map = decode(input_item, content_type)
    parse_text_inputs_params(request_input, input_item, input_map, **kwargs)
    parse_adapters(request_input, input_item, input_map, **kwargs)
    return request_input


def parse_text_inputs_params(request_input: TextInput, input_item: Input,
                             input_map: Dict, **kwargs):
    invoke_type = input_item.get_property("X-Amzn-SageMaker-Forwarded-Api")
    tokenizer = kwargs.get("tokenizer")
    image_token = kwargs.get("image_placeholder_token")
    configs = kwargs.get("configs")
    is_rolling_batch = kwargs.get("is_rolling_batch", False)
    is_bedrock = False
    if configs is not None:
        is_bedrock = configs.bedrock_compat
    if is_chat_completions_request(input_map):
        inputs, param = parse_chat_completions_request(
            input_map,
            kwargs.get("is_rolling_batch"),
            tokenizer,
            image_token=image_token,
            configs=configs)
    elif is_bedrock:
        inputs, param = parse_3p_request(input_map,
                                         kwargs.get("is_rolling_batch"),
                                         tokenizer, invoke_type)
    elif is_rolling_batch:
        inputs, param = parse_lmi_default_request_rolling_batch(input_map)
    else:
        inputs = input_map.pop("inputs", input_map)
        param = input_map.pop("parameters", {})

    request_input.input_text = inputs
    request_input.parameters = param

    # TODO: Instead of modifying user parameters, maintain this in server_parameters.
    #  Added here for backward compatibility
    # re-organize the parameters
    if "cached_prompt" in input_map:
        request_input.parameters["cached_prompt"] = input_map.pop(
            "cached_prompt")


def add_server_maintained_params(request_input: RequestInput,
                                 input_item: Input, **kwargs):
    """
    Add some additional parameters for djl serving to do some work that are necessary.

    :param request_input: request_input
    :param input_item: Input object
    :param kwargs: other parameters that are needed.
    """
    request_input.server_parameters = request_input.parameters.copy()
    # Per request streaming is only supported by rolling batch
    if "seed" not in request_input.server_parameters:
        # set server provided seed if seed is not part of request
        if input_item.contains_key("seed"):
            request_input.server_parameters["seed"] = input_item.get_as_string(
                key="seed")

    # setting the output formatter
    if not "output_formatter" in request_input.server_parameters:
        request_input.server_parameters["output_formatter"] = kwargs.get(
            "configs").output_formatter

    output_formatter = request_input.server_parameters["output_formatter"]
    if output_formatter == "json" or output_formatter == "sse":
        request_input.tgi_compat = kwargs.get("configs").tgi_compat


def parse_adapters(request_input: TextInput, input_item: Input,
                   input_map: Dict, **kwargs):
    adapter_registry = kwargs.get("adapter_registry")
    # if adapter registry exists and not empty, then we assume, peft is supported for the incoming
    if adapter_registry:
        input_len = len(request_input.input_text) if isinstance(
            request_input.input_text, list) else 1
        adapters_per_item = _fetch_adapters_from_input(input_map, input_item)
        if adapters_per_item:
            _validate_adapters(adapters_per_item,
                               kwargs.get("adapter_registry"))
        else:
            # inference with just base model.
            adapters_per_item = [""] * input_len

        if input_len != len(adapters_per_item):
            raise ValueError(
                f"Number of adapters is not equal to the number of inputs")
        # lookup the adapter registry to get the adapter details of the registered adapter.
        adapters_data = [
            kwargs.get("adapter_registry").get(adapter, None)
            for adapter in adapters_per_item
        ]
        if len(adapters_data) == 1:
            adapters_data = adapters_data[0]

        request_input.adapters = adapters_data


def _fetch_adapters_from_input(input_map: dict, input_item: Input):
    adapters_per_item = []
    if "adapters" in input_map:
        adapters_per_item = input_map.pop("adapters", [])

    # check content, possible in workflow approach
    if input_item.contains_key("adapter"):
        adapters_per_item = input_item.get_as_string("adapter")

    # check properties, possible from header
    if "adapter" in input_item.get_properties():
        adapters_per_item = input_item.get_properties()["adapter"]

    if not isinstance(adapters_per_item, list):
        adapters_per_item = [adapters_per_item]

    return adapters_per_item


def _validate_adapters(adapters_per_item, adapter_registry):
    for adapter_name in adapters_per_item:
        if adapter_name and adapter_name not in adapter_registry:
            raise ValueError(f"Adapter {adapter_name} is not registered")


def parse_lmi_default_request_rolling_batch(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid request payload. Request payload should be a json object specifying the 'inputs' field. Received payload {payload}"
        )

    inputs = payload.get("inputs", None)
    if inputs is None:
        raise ValueError(
            f"Invalid request payload. Request payload should be a json object specifying the 'inputs' field. Received payload {payload}"
        )

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Invalid request payload. 'parameters' must be provided as an object of key-value pairs. Received payload {payload}"
        )

    parameters["stream"] = payload.get("stream", False)
    return inputs, parameters
