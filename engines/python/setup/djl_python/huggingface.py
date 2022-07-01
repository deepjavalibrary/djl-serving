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

import importlib
import logging
import os
import sys
import time
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import encode, decode
from transformers import pipeline

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
}


class HuggingFaceService(object):
    def __init__(self):
        self.hf_pipeline = None
        self.initialized = False

    def initialize(self, properties: dict):
        model_dir = properties.get("model_dir")
        device_id = int(properties.get("device_id"))
        model_id = properties.get("model_id")
        task = properties.get("task")
        if task:
            self.hf_pipeline = self.get_pipeline(task=task,
                                                 model_id=model_id,
                                                 model_dir=model_dir,
                                                 device=device_id)
        elif "config.json" in os.listdir(model_dir):
            self.hf_task = infer_task_from_model_architecture(
                f"{model_dir}/config.json")
            self.hf_pipeline = self.get_pipeline(task=hf_task,
                                                 model_id=model_id,
                                                 model_dir=model_dir,
                                                 device=device_id)
        else:
            raise ValueError("You need to define 'task' options.")

        self.initialized = True

    def inference(self, inputs):
        outputs = Output()
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

            encode(outputs, prediction, accept)
            return outputs
        except Exception as e:
            logging.error(e, exc_info=True)
            # error handling
            outputs.set_code(500)
            outputs.set_message(str(e))
            outputs.add("inference failed", key="data")

        return outputs

    @staticmethod
    def get_pipeline(task: str, device: int, model_id, model_dir: str,
                     **kwargs):
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
            hf_pipeline = wrap_conversation_pipeline(hf_pipeline)

        return hf_pipeline

    @staticmethod
    def wrap_conversation_pipeline(pipeline):
        def wrapped_pipeline(inputs, *args, **kwargs):
            converted_input = Conversation(
                inputs["text"],
                past_user_inputs=inputs.get("past_user_inputs", []),
                generated_responses=inputs.get("generated_responses", []),
            )
            prediction = pipeline(converted_input, *args, **kwargs)
            return {
                "generated_text": prediction.generated_responses[-1],
                "conversation": {
                    "past_user_inputs": prediction.past_user_inputs,
                    "generated_responses": prediction.generated_responses,
                },
            }

        return wrapped_pipeline

    @staticmethod
    def infer_task_from_model_architecture(model_config_path: str,
                                           architecture_index=0):
        with open(model_config_path, "r+") as config_file:
            config = json.loads(config_file.read())
            architecture = config.get("architectures",
                                      [None])[architecture_index]

        task = None
        for arch_options in ARCHITECTURES_2_TASK:
            if architecture.endswith(arch_options):
                task = ARCHITECTURES_2_TASK[arch_options]

        if task is None:
            raise ValueError(
                f"Task couldn't be inferenced from {architecture}. Please manually set `task` option."
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
