#!/usr/bin/env python
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import json
import logging
import os

import torch
from transformers import pipeline, Conversation, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from djl_python.encode_decode import encode, decode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.streaming_utils import StreamingUtils

ARCHITECTURES_2_TASK = {
    "TapasForQuestionAnswering": "table-question-answering",
    "ForQuestionAnswering": "question-answering",
    "ForTokenClassification": "token-classification",
    "ForSequenceClassification": "text-classification",
    "ForMultipleChoice": "multiple-choice",
    "ForMaskedLM": "fill-mask",
    "ForCausalLM": "text-generation",
    "ForConditionalGeneration": "text2text-generation",
    "MTModel": "text2text-generation",
    "EncoderDecoderModel": "text2text-generation",
    # Model specific task for backward comp
    "GPT2LMHeadModel": "text-generation",
    "T5WithLMHeadModel": "text2text-generation",
    "BloomModel": "text-generation",
}


def get_torch_dtype_from_str(dtype: str):
    if dtype == "auto":
        return dtype
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "int8":
        return torch.int8
    if dtype is None:
        return None
    raise ValueError(f"Invalid data type: {dtype}")


class HuggingFaceService(object):

    def __init__(self):
        self.hf_pipeline = None
        self.initialized = False
        self.enable_streaming = False
        self.model = None
        self.tokenizer = None

    def initialize(self, properties: dict):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        device_id = int(properties.get("device_id", "-1"))
        task = properties.get("task")
        tp_degree = int(properties.get("tensor_parallel_degree", "-1"))
        self.enable_streaming = properties.get("enable_streaming",
                                               "false").lower() == "true"
        # HF Acc handling
        kwargs = {}
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        if "device_map" in properties:
            kwargs["device_map"] = properties.get("device_map")
            logging.info(f"Using device map {kwargs['device_map']}")
        elif tp_degree > 0:
            kwargs["device_map"] = "auto"
            world_size = torch.cuda.device_count()
            assert world_size == tp_degree, f"TP degree ({tp_degree}) doesn't match available GPUs ({world_size})"
            logging.info(f"Using {world_size} gpus")
        if "load_in_8bit" in properties:
            if "device_map" not in kwargs:
                raise ValueError(
                    "device_map should set when load_in_8bit is set")
            kwargs["load_in_8bit"] = properties.get("load_in_8bit")
        if "low_cpu_mem_usage" in properties:
            kwargs["low_cpu_mem_usage"] = properties.get("low_cpu_mem_usage")

        if "dtype" in properties:
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties.get("dtype"))
        if self.enable_streaming:
            self._init_model_and_tokenizer(model_id_or_path, **kwargs)
            self.initialized = True
            return

        if not task:
            task = self.infer_task_from_model_architecture(model_id_or_path)

        self.hf_pipeline = self.get_pipeline(task=task,
                                             model_id_or_path=model_id_or_path,
                                             device=device_id,
                                             kwargs=kwargs)

        self.initialized = True

    def inference(self, inputs):
        try:
            content_type = inputs.get_property("Content-Type")
            accept = inputs.get_property("Accept")
            if not accept:
                accept = content_type if content_type.startswith(
                    "tensor/") else "application/json"
            elif accept == "*/*":
                accept = "application/json"

            input_map = decode(inputs, content_type)
            data = input_map.pop("inputs", input_map)
            parameters = input_map.pop("parameters", {})
            outputs = Output()

            if self.enable_streaming:
                stream_generator = StreamingUtils.get_stream_generator(
                    "Accelerate")
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, data,
                                     **parameters))
                return outputs

            prediction = self.hf_pipeline(data, **parameters)

            encode(outputs, prediction, accept)
        except Exception as e:
            logging.exception("Huggingface inference failed")
            # error handling
            outputs = Output().error(str(e))

        return outputs

    def get_pipeline(self, task: str, device: int, model_id_or_path: str,
                     kwargs):
        # define tokenizer or feature extractor as kwargs to load it the pipeline correctly
        if task in {
                "automatic-speech-recognition",
                "image-segmentation",
                "image-classification",
                "audio-classification",
                "object-detection",
                "zero-shot-image-classification",
        }:
            kwargs["feature_extractor"] = model_id_or_path
        else:
            kwargs["tokenizer"] = model_id_or_path

        use_pipeline = True
        for element in ["load_in_8bit", "low_cpu_mem_usage"]:
            if element in kwargs:
                use_pipeline = False
        # build pipeline
        if use_pipeline:
            if "device_map" in kwargs:
                hf_pipeline = pipeline(task=task,
                                       model=model_id_or_path,
                                       **kwargs)
            else:
                hf_pipeline = pipeline(task=task,
                                       model=model_id_or_path,
                                       device=device,
                                       **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
            kwargs.pop("tokenizer", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, **kwargs)
            hf_pipeline = pipeline(task=task, model=model, tokenizer=tokenizer)

        # wrap specific pipeline to support better ux
        if task == "conversational":
            hf_pipeline = self.wrap_conversation_pipeline(hf_pipeline)

        if task == "text-generation":
            hf_pipeline.tokenizer.padding_side = "left"
            if not hf_pipeline.tokenizer.pad_token:
                hf_pipeline.tokenizer.pad_token = hf_pipeline.tokenizer.eos_token
            hf_pipeline = self.wrap_text_generation_pipeline(hf_pipeline)

        return hf_pipeline

    def _init_model_and_tokenizer(self, model_id_or_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left")
        model_config = AutoConfig.from_pretrained(model_id_or_path)
        architectures = model_config.architectures
        if architectures and architectures[0].endswith(
                "ForConditionalGeneration"):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id_or_path, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, **kwargs)

    @staticmethod
    def wrap_conversation_pipeline(hf_pipeline):

        def wrapped_pipeline(inputs, *args, **kwargs):
            converted_input = Conversation(
                inputs["text"],
                past_user_inputs=inputs.get("past_user_inputs", []),
                generated_responses=inputs.get("generated_responses", []),
            )
            prediction = hf_pipeline(converted_input, *args, **kwargs)
            return {
                "generated_text": prediction.generated_responses[-1],
                "conversation": {
                    "past_user_inputs": prediction.past_user_inputs,
                    "generated_responses": prediction.generated_responses,
                },
            }

        return wrapped_pipeline

    @staticmethod
    def wrap_text_generation_pipeline(hf_pipeline):

        def wrapped_pipeline(inputs, *args, **kwargs):
            model = hf_pipeline.model
            tokenizer = hf_pipeline.tokenizer
            input_tokens = tokenizer(inputs, padding=True,
                                     return_tensors="pt").to(
                                         torch.cuda.current_device())
            with torch.no_grad():
                output_tokens = model.generate(*args, **input_tokens, **kwargs)
            generated_text = tokenizer.batch_decode(output_tokens,
                                                    skip_special_tokens=True)

            return [{"generated_text": s} for s in generated_text]

        return wrapped_pipeline

    @staticmethod
    def infer_task_from_model_architecture(model_config_path: str):
        model_config = AutoConfig.from_pretrained(model_config_path)
        architecture = model_config.architectures[0]

        task = None
        for arch_options in ARCHITECTURES_2_TASK:
            if architecture.endswith(arch_options):
                task = ARCHITECTURES_2_TASK[arch_options]

        if task is None:
            raise ValueError(
                f"Task couldn't be inferred from {architecture}. Please manually set `task` option."
            )
        return task


_service = HuggingFaceService()


def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
