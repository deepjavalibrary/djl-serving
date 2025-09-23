import json
import os

input_payload_options = [
    {"key": "inputPrompt", "target": "inputs"},
    {"key": "temperature", "target": "temperature"},
    {"key": "top_p", "target": "top_p"},
    {"key": "frequency_penalty", "target": "frequency_penalty"},
    {"key": "top_k", "target": "top_k"},
    {"key": "maxOutputToken", "target": "request_output_len"},  # Used by trtllm
    {"key": "maxOutputToken", "target": "max_new_tokens"},  # Used by vLLM
]

parameters_options = [
    {"key": "maxOutputToken", "target": "max_new_tokens"},
    {"key": "temperature", "target": "temperature"},
    {"key": "top_p", "target": "top_p"},
    {"key": "frequency_penalty", "target": "frequency_penalty"},
    {"key": "top_k", "target": "top_k"},
    {"key": "do_sample", "target": "do_sample"},
    {"key": "stream", "target": "stream"},
    {"key": "stop", "target": "stop"},
    {"key": "return_logprobs", "target": "return_logprobs"},
    {"key": "details", "target": "details"},
    {"key": "decoder_input_details", "target": "decoder_input_details"},
    {"key": "seed", "target": "seed"},
    {"key": "echo", "target": "echo"},
    {"key": "adapters", "target": "adapters"}
]
additional_properties_options = []


class ProcessingClass:
    def __init__(self):
        pass

    def pre_process(self, input_payload, tokenizer):
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
                input_dict[input_payload_option["target"]] = input_payload[input_payload_option["key"]]

        if "parameters" in input_payload:
            input_dict["parameters"] = {}
            for parameters_option in parameters_options:
                if parameters_option["key"] in input_payload["parameters"]:
                    input_dict["parameters"][parameters_option["target"]] = input_payload["parameters"][parameters_option["key"]]
        if "additionalProperties" in input_payload:
            for additional_properties_option in additional_properties_options:
                if additional_properties_option["key"] in input_payload["additionalProperties"]:
                    input_dict[additional_properties_option["target"]] = input_payload["additionalProperties"][
                        additional_properties_option["key"]
                    ]

        return input_dict

processing = ProcessingClass()


def pre_process(input_payload, tokenizer, *args, **kwargs):
    """
    Pre-process the input payload.

    Args:
        input_payload (dict): The input payload received from client
        tokenizer: The tokenizer

    Returns:
        (dict): The input payload after preprocessing
    """
    return processing.pre_process(input_payload, tokenizer)
