import logging
import os
import torch

import tensorrt_llm_toolkit
from tensorrt_llm_toolkit.utils import utils as toolkit_utils

from transformers import AutoConfig

from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.encode_decode import encode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.utils import parse_input_with_formatter, InputFormatConfigs


def _get_value_based_on_tensor(value, index=None):
    if isinstance(value, torch.Tensor):
        if index:
            return value.cpu().numpy()[index]
        else:
            return value.cpu().item()
    else:
        return value


def _get_generation_result_from_python_backend(generations, inputs_size):
    batch_size = sum(inputs_size)
    tokens_results = [[] for _ in range(batch_size)
                      ]  # list[list], [batch_size, generated_tokens_len]
    prediction_results = [{} for _ in range(batch_size)
                          ]  # list[dict], [batch_size]
    cum_log_probs = [0.0
                     for _ in range(batch_size)]  # list[dict], [batch_size]
    for generation in generations:  # each token of whole batch
        for i in range(len(generation)):  # loop through each batch
            # generation will be none, when it is already finished for that input
            if not generation[i]:
                continue
            # generated_text will not be none, only during the last token.
            if generation[i].generated_text:
                result = {
                    "generated_text": generation[i].generated_text,
                    'details': {
                        # TODO: add finish reason
                        "tokens": tokens_results[i]
                    }
                }
                prediction_results[i] = result
            else:
                curr_cum_log_prob = _get_value_based_on_tensor(
                    generation[i].cum_logprob)
                log_prob = curr_cum_log_prob - cum_log_probs[i]
                token_result = {
                    'id':
                    _get_value_based_on_tensor(generation[i].token_id,
                                               index=0),
                    'text':
                    generation[i].token_text,
                    'log_prob':
                    log_prob if i < len(tokens_results) else curr_cum_log_prob,
                }
                cum_log_probs[i] = curr_cum_log_prob
                tokens_results[i].append(token_result)
    return prediction_results


def _get_accept_and_content_type(batch_item) -> tuple[str, str]:
    content_type = batch_item.get_property("Content-Type")
    accept = batch_item.get_property("Accept")
    if not accept:
        content_type = content_type if content_type else "application/json"
        accept = content_type if content_type.startswith(
            "tensor/") else "application/json"
    elif "*/*" in accept:
        accept = "application/json"
    return content_type, accept


class TRTLLMPythonService:

    PYTHON_BACKEND_SUPPORTED_MODELS = {'t5'}

    def __init__(self):
        self.model = None
        self.trt_configs = None
        self.initialized = False
        self.is_client_side_batch = []
        self.input_formatter = None

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)
        self._load_model(properties)
        self.input_formatter = InputFormatConfigs(
            is_rolling_batch=False,
            is_adapters_supported=False,
            output_formatter=self.trt_configs.output_formatter,
            tokenizer=self.model.tokenizer)
        self.initialized = True
        return

    def parse_input(
        self, inputs: Input, tokenizer, output_formatter
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
        parsed_input = parse_input_with_formatter(
            inputs, input_format_configs=self.input_formatter)
        self.is_client_side_batch = parsed_input.is_client_side_batch
        return parsed_input.input_data, parsed_input.input_size, parsed_input.parameters, parsed_input.errors, parsed_input.batch

    def inference(self, inputs: Input) -> Output:
        """
        Does preprocessing and sends new requests to the rolling batch script for inference

        :param inputs: (Input) a batch of inputs, each corresponding to a new request

        :return outputs (Output): a batch of outputs that contain status code, output text, and other information
        """
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs, None, self.trt_configs.output_formatter)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                outputs.add(err, key="data", batch_index=i)
            return outputs

        params = parameters[0]

        if "output_formatter" in params:
            # output formatter is not supported for TensorRT-LLM python backend.
            params.pop("output_formatter")
        if "stream" in params:
            # TensorRT-LLM python backend handler does not support streaming yet.
            params.pop("stream")
        if params.get("details", False):
            return self._stream_inference(inputs, input_data, input_size,
                                          params, batch)

        detokenized_python_response = self.model.generate(input_data, **params)
        results = [{
            "generated_text": s
        } for s in detokenized_python_response.batch_generation()]
        offset = 0
        for i, item in enumerate(batch):
            content_type, accept = _get_accept_and_content_type(item)
            batch_item = results[offset:offset + input_size[i]] if i < len(
                self.is_client_side_batch
            ) and self.is_client_side_batch[i] else results[offset]
            encode(outputs,
                   batch_item,
                   accept,
                   key=inputs.get_content().key_at(i))
            offset += input_size[i]
        return outputs

    def _load_model(self, properties):
        model_config = self._get_config(properties)
        if model_config.model_type in self.PYTHON_BACKEND_SUPPORTED_MODELS:
            self.model = tensorrt_llm_toolkit.init_inference(
                self.trt_configs.model_id_or_path,
                **properties,
                use_python_backend=True)
        else:
            raise ValueError(
                f"You cannot disable rolling batch if its not any of these models"
                f" {self.PYTHON_BACKEND_SUPPORTED_MODELS}. Please enable it with auto or trtllm "
                f"values to option.rolling_batch")

    def _get_config(self, properties):
        model_path = self.trt_configs.model_id_or_path
        if not os.path.isfile(os.path.join(model_path, 'config.json')):
            model_path = toolkit_utils.get_python_backend_engine_path(
                model_path, properties)
            if not os.path.isfile(os.path.join(model_path, 'config.json')):
                raise ValueError(
                    f"Could not find config.json in {self.trt_configs.model_id_or_path} or"
                    f"{model_path} for TensorRT python backend")

        return AutoConfig.from_pretrained(
            model_path, trust_remote_code=self.trt_configs.trust_remote_code)

    # TODO TrtLLM python backend: Change it once T5 bug is fixed.
    def _stream_inference(self, inputs: Input, input_data: list[str],
                          input_size: list[int], parameters: dict,
                          batch: list) -> Output:
        outputs = Output()
        detokenized_python_response = self.model.generate(
            input_data, **parameters)
        generations = detokenized_python_response.stream_batch_generation()
        results = _get_generation_result_from_python_backend(
            generations, input_size)
        offset = 0
        for i, item in enumerate(batch):
            item = batch[i]
            accept, content_type = _get_accept_and_content_type(item)
            batch_item = results[offset:offset + input_size[i]] if i < len(
                self.is_client_side_batch
            ) and self.is_client_side_batch[i] else results[offset]
            encode(outputs,
                   batch_item,
                   accept,
                   key=inputs.get_content().key_at(i))
            offset += input_size[i]
        return outputs
