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

from transformers import pipeline, Conversation

from djl_python.encode_decode import encode, decode
from djl_python.inputs import Input
from djl_python.outputs import Output

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


class HuggingFaceService(object):

    def __init__(self):
        self.hf_pipeline = None
        self.initialized = False

    def initialize(self, properties: dict):
        model_dir = properties.get("model_dir")
        device_id = int(properties.get("device_id", "-1"))
        model_id = properties.get("model_id")
        task = properties.get("task")
        # HF Acc handling
        kwargs = {
            "load_in_8bit": bool(properties.get("load_in_8bit", "FALSE")),
            "low_cpu_mem_usage": bool(properties.get("low_cpu_mem_usage", "TRUE")),
        }
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        if "device_map" in properties:
            kwargs["device_map"] = properties.get("device_map")
        if "dtype" in properties:
            kwargs["torch_dtype"] = properties.get("dtype")
        if task:
            self.hf_pipeline = self.get_pipeline(task=task,
                                                 model_id=model_id,
                                                 model_dir=model_dir,
                                                 device=device_id,
                                                 kwargs=kwargs)
        elif "config.json" in os.listdir(model_dir):
            task = self.infer_task_from_model_architecture(
                f"{model_dir}/config.json")
            self.hf_pipeline = self.get_pipeline(task=task,
                                                 model_id=model_id,
                                                 model_dir=model_dir,
                                                 device=device_id,
                                                 kwargs=kwargs)
        else:
            raise ValueError("You need to define 'task' options.")

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
            parameters = input_map.pop("parameters", None)

            # pass inputs with all kwargs in data
            if parameters is not None:
                prediction = self.hf_pipeline(data, **parameters)
            else:
                prediction = self.hf_pipeline(data)

            outputs = Output()
            encode(outputs, prediction, accept)
        except Exception as e:
            logging.exception("Huggingface inference failed")
            # error handling
            outputs = Output().error(str(e))

        return outputs

    def get_pipeline(self, task: str, device: int, model_id: str,
                     model_dir: str, **kwargs):
        model = model_id if model_id else model_dir

        # define tokenizer or feature extractor as kwargs to load it the pipeline correctly
        if task in {
                "automatic-speech-recognition",
                "image-segmentation",
                "image-classification",
                "audio-classification",
                "object-detection",
                "zero-shot-image-classification",
        }:
            kwargs["feature_extractor"] = model
        else:
            kwargs["tokenizer"] = model

        # load pipeline
        hf_pipeline = pipeline(task=task,
                               model=model,
                               device=device,
                               framework="pt",
                               **kwargs)

        # wrapp specific pipeline to support better ux
        if task == "conversational":
            hf_pipeline = self.wrap_conversation_pipeline(hf_pipeline)

        return hf_pipeline

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
    def infer_task_from_model_architecture(model_config_path: str):
        with open(model_config_path, "r+") as config_file:
            config = json.loads(config_file.read())
            architecture = config.get("architectures", [None])[0]

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
