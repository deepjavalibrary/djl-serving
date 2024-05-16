import logging
from typing import Union, Callable, Any, List

from djl_python.inputs import Input
from djl_python.encode_decode import decode
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request
from dataclasses import dataclass, field


@dataclass
class ParsedInput:
    input_data: list[str]
    input_size: list[int]
    parameters: list[dict]
    errors: dict
    batch: list
    is_client_side_batch: list = field(default_factory=lambda: [])
    adapters: list = None
    found_adapters: bool = False


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
            input_map = decode(item, content_type)
            _inputs, _param, is_client_side_batch[i] = _parse_inputs_params(
                input_map, item, input_format_configs)
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

        adapters.extend(adapters_per_item)
        found_adapters = found_adapter_per_item or found_adapters

        for _ in range(input_size[i]):
            parameters.append(_param)

    return ParsedInput(input_data=input_data,
                       input_size=input_size,
                       parameters=parameters,
                       errors=errors,
                       batch=batch,
                       is_client_side_batch=is_client_side_batch,
                       adapters=adapters,
                       found_adapters=found_adapters)


def _parse_inputs_params(input_map, item, input_format_configs):
    if is_chat_completions_request(input_map):
        _inputs, _param = parse_chat_completions_request(
            input_map, input_format_configs.is_rolling_batch,
            input_format_configs.tokenizer)
    else:
        _inputs = input_map.pop("inputs", input_map)
        _param = input_map.pop("parameters", {})

    # Add some additional parameters that are necessary.
    # Per request streaming is only supported by rolling batch
    if input_format_configs.is_rolling_batch:
        _param["stream"] = input_map.pop("stream", False)

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
