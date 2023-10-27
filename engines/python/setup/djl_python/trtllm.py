import json
import logging
import os
import re

import torch
from transformers import (
    pipeline, Pipeline, Conversation, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, StoppingCriteria, StoppingCriteriaList)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from djl_python.rolling_batch.trtllm_rolling_batch import TRTLLMRollingBatch
from djl_python.encode_decode import encode, decode
from djl_python.inputs import Input
from djl_python.outputs import Output

class TRTLLMService(object):
    def __init__(self):
        self.initialized = False
        self.tensor_parallel_degree = -1
        self.pipeline_parallel_degree = -1
        self.dtype = None
        self.model_id_or_path = None
        self.device = None
        self.model = None
        self.device = None
        self.tokenizer = None
        self.model_config = None
        self.rolling_batch = None


    def initialize(self, properties):
        self.tensor_parallel_degree = int(properties.get("tensor_parallel_degree", 1))
        self.pipeline_parallel_degree = int(properties.get("pipeline_parallel_degree", 1))
        self.dtype = properties.get("data_type", self.dtype)
        self.dtype = properties.get("dtype", "fp32")
        self.model_id_or_path = properties.get("model_id") or properties.get("model_dir")
        device_id = int(properties.get("device_id", "-1"))
        self.device = f"cuda:{device_id}" if device_id >= 0 else None
        kwargs = {}
        if "revision" in properties:
            kwargs["revision"] = properties.get('revision')
        if "output_formatter" in properties:
            kwargs["output_formatter"] = properties.get("output_formatter")
        if "waiting_steps" in properties:
            kwargs["waiting_steps"] = int(properties.get("waiting_steps"))
        is_mpi = properties.get("engine") != "Python"
        if is_mpi:
            self.device = int(os.getenv("LOCAL_RANK", 0))
        self.rolling_batch = TRTLLMRollingBatch(self.model_id_or_path,
                                                    self.device, properties,
                                                    **kwargs)
        self.initialized = True

    # def parse_input(self, inputs):
    #   """
    #   Get input data from Input object into a list of ? (unknown) objects
    #   From fastertransformer inference
    #   """
    #   input_data = []
    #   parameters = {}
    #   batch = inputs.get_batches()
    #   for item in batch:
    #       input_map = item.get_as_json()
    #       input_text = input_map.pop("inputs", input_map)
    #       if isinstance(input_text, str):
    #             input_text = [input_text]
    #         input_data.extend(input_text)
    #         if first:
    #           parameters = input_map.pop("parameters", {})
    #           first = False
    #         else:
    #             if parameters != input_map.pop("parameters", {}):
    #                 return Output().error("In order to enable dynamic batching, all input batches must have the same parameters")

    #     return input_data

    def parse_input(self, inputs):
        input_data = []
        input_size = []
        parameters = []
        adapters = []
        errors = {}
        batch = inputs.get_batches()
        first = True
        for i, item in enumerate(batch):
            try:
                content_type = item.get_property("Content-Type")
                input_map = decode(item, content_type)
                _inputs = input_map.pop("inputs", input_map)
                # remove adapters
                #adapters_per_item = self._fetch_adapters_from_input(input_map, item)
                if first:
                    parameters.append(input_map.pop("parameters", {}))
                    first = False
                else:
                    param = input_map.pop("parameters", {})

                if not isinstance(_inputs, list):
                    _inputs = [_inputs]

                # if not isinstance(adapters_per_item, list):
                #     adapters_per_item = [adapters_per_item]

                # if not adapters_per_item:
                #     ## inference with just base model.
                #     adapters_per_item = [""] * len(_inputs)
                # else:
                #     if len(_inputs) != len(adapters_per_item):
                #         ## input_size list needs to be appended as it's used during output processing
                #         input_size.append(0)
                #         raise Exception(
                #             "Number of adapters is not equal to the number of inputs"
                #         )

                input_data.extend(_inputs)
                input_size.append(len(_inputs))
                # adapters.extend(adapters_per_item)

                if "cached_prompt" in input_map:
                    parameters[i]["cached_prompt"] = input_map.pop(
                        "cached_prompt")

                seed_key = 'seed' if inputs.is_batch() else f'batch_{i}.seed'
                if item.contains_key(seed_key):
                    seed = parameters[i].get("seed")
                    if not seed:
                        # set server provided seed if seed is not part of request
                        parameters[i]["seed"] = item.get_as_string(
                            key=seed_key)
            except Exception as e:  # pylint: disable=broad-except
                logging.exception(f"Parse input failed: {i}")
                errors[i] = str(e)

        return input_data, input_size, adapters, parameters, errors, batch

    def inference(self, inputs):
        """
        Run inference - wrapper around rolling batch call
        """
        outputs = Output()
        input_data, input_size, adapters, parameters, errors, batch = self.parse_input(inputs)

        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                err = json.dumps({"code": 424, "error": err})
                err = json.dumps({"data": err, "last": True})
                outputs.add(err, key="data", batch_index=i)
            return outputs

        if inputs.get_property("reset_rollingbatch"):
            self.rolling_batch.reset()
        
        result = self.rolling_batch.inference(input_data, parameters)

        for i in range(len(batch)):
            err = errors.get(i)
            if err:
                err = json.dumps({"code": 424, "error": err})
                err = json.dumps({"data": err, "last": True})
                outputs.add(err, key="data", batch_index=i)
            else:
                outputs.add(result[idx], key="data", batch_index=i)
                idx += 1

        content_type = self.rolling_batch.get_content_type()
        if content_type:
            outputs.add_property("content-type", content_type)
        return outputs

_service = TRTLLMService()

def handle(inputs):
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)
