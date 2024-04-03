import logging
from djl_python.inputs import Input
from djl_python.encode_decode import encode, decode
from djl_python.chat_completions.chat_utils import is_chat_completions_request, parse_chat_completions_request


def parse_input_with_client_batch(
    inputs: Input, tokenizer, output_formatter
) -> tuple[list[str], list[int], list[dict], dict, list, list]:
    """
    Preprocessing function that extracts information from Input objects.

    :param output_formatter: output formatter for the request
    :param inputs :(Input) a batch of inputs, each corresponding to a new request
    :param tokenizer: the tokenizer used for inference

    :return input_data (list[str]): a list of strings, each string being the prompt in a new request
    :return input_size (list[int]): a list of ints being the size of each new request
    :return parameters (list[dict]): parameters pertaining to each request
    :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
    :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
    :return is_client_size_batch (list): list of boolean value representing whether the input is a client side batch
    """
    input_data = []
    input_size = []
    parameters = []
    errors = {}
    batch = inputs.get_batches()
    # only for dynamic batch
    is_client_size_batch = [False for _ in range(len(batch))]
    for i, item in enumerate(batch):
        try:
            content_type = item.get_property("Content-Type")
            input_map = decode(item, content_type)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"Parse input failed: {i}")
            input_size.append(0)
            errors[i] = str(e)
            continue

        if is_chat_completions_request(input_map):
            _inputs, _param = parse_chat_completions_request(
                input_map, True, tokenizer)
        else:
            _inputs = input_map.pop("inputs", input_map)
            _param = input_map.pop("parameters", {})
            _param["stream"] = input_map.pop("stream", False)
        if not isinstance(_inputs, list):
            _inputs = [_inputs]
        else:
            is_client_size_batch[i] = True
        input_data.extend(_inputs)
        input_size.append(len(_inputs))

        if "cached_prompt" in input_map:
            _param["cached_prompt"] = input_map.pop("cached_prompt")
        if "seed" not in _param:
            # set server provided seed if seed is not part of request
            if item.contains_key("seed"):
                _param["seed"] = item.get_as_string(key="seed")
        if not "output_formatter" in _param:
            _param["output_formatter"] = output_formatter

        for _ in range(input_size[i]):
            parameters.append(_param)

    return input_data, input_size, parameters, errors, batch, is_client_size_batch


def parse_input(
        inputs: Input, tokenizer, output_formatter
) -> tuple[list[str], list[int], list[dict], dict, list]:
    """
    Preprocessing function that extracts information from Input objects.

    :param output_formatter: output formatter for the request
    :param inputs :(Input) a batch of inputs, each corresponding to a new request
    :param tokenizer: the tokenizer used for inference

    :return input_data (list[str]): a list of strings, each string being the prompt in a new request
    :return input_size (list[int]): a list of ints being the size of each new request
    :return parameters (list[dict]): parameters pertaining to each request
    :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
    :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
    """
    input_data, input_size, parameters, errors, batch, _ = parse_input_with_client_batch(
        inputs, tokenizer, output_formatter)
    return input_data, input_size, parameters, errors, batch
