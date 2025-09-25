import json
import os

input_payload_options = [
    {
        "key": "inputPrompt",
        "target": "inputs"
    },
    {
        "key": "temperature",
        "target": "temperature"
    },
    {
        "key": "top_p",
        "target": "top_p"
    },
    {
        "key": "frequency_penalty",
        "target": "frequency_penalty"
    },
    {
        "key": "top_k",
        "target": "top_k"
    },
    {
        "key": "maxOutputToken",
        "target": "request_output_len"
    },  # Used by trtllm
    {
        "key": "maxOutputToken",
        "target": "max_new_tokens"
    },  # Used by vLLM
]

parameters_options = [{
    "key": "maxOutputToken",
    "target": "max_new_tokens"
}, {
    "key": "temperature",
    "target": "temperature"
}, {
    "key": "top_p",
    "target": "top_p"
}, {
    "key": "frequency_penalty",
    "target": "frequency_penalty"
}, {
    "key": "top_k",
    "target": "top_k"
}, {
    "key": "do_sample",
    "target": "do_sample"
}, {
    "key": "stream",
    "target": "stream"
}, {
    "key": "stop",
    "target": "stop"
}, {
    "key": "return_logprobs",
    "target": "return_logprobs"
}, {
    "key": "details",
    "target": "details"
}, {
    "key": "decoder_input_details",
    "target": "decoder_input_details"
}, {
    "key": "seed",
    "target": "seed"
}, {
    "key": "echo",
    "target": "echo"
}, {
    "key": "adapters",
    "target": "adapters"
}]


def pre_process(input_payload, tokenizer):
    """
    Pre-process the input payload.

    Args:
        input_payload (dict): The input payload received from client
        tokenizer: The tokenizer

    Returns:
        (dict): The input payload after preprocessing
    """
    input_dict = {}

    for input_payload_option in input_payload_options:
        if input_payload_option["key"] in input_payload:
            input_dict[input_payload_option["target"]] = input_payload[
                input_payload_option["key"]]

    if "parameters" in input_payload:
        input_dict["parameters"] = {}
        for parameters_option in parameters_options:
            if parameters_option["key"] in input_payload["parameters"]:
                input_dict["parameters"][
                    parameters_option["target"]] = input_payload["parameters"][
                        parameters_option["key"]]

    return input_dict


def post_process(result):
    """
    Processes the output from the model and returns the response

    Args:
        result (dict): The output from the model

    Returns:
        (str): json dumps of output after postprocessing, consistent with
               chat completion format
    """
    output_dict = {
        "error_message":
        "",
        "completions": [{
            "completionText": result["decoded_text"],
            "token_logprobs": result.get("token_logprobs", []),
            "top_logprobs": [],
            "tokens": result.get("tokens", ""),
            "text_offsets": -1,
            "model_version": os.getenv("model_endpoint_version"),
        }],
    }

    return json.dumps(output_dict)
