#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import torch
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    pipeline,
    Conversation,
    SquadExample,
)
import transformers
from deepspeed.module_inject.replace_policy import (
    HFBertLayerPolicy,
    HFGPTNEOLayerPolicy,
    GPTNEOXLayerPolicy,
    HFGPTJLayerPolicy,
    MegatronLayerPolicy,
    HFGPT2LayerPolicy,
    BLOOMLayerPolicy,
    HFOPTLayerPolicy,
    HFCLIPLayerPolicy,
)
import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional

SUPPORTED_MODEL_TYPES = {
    "roberta",
    "gpt2",
    "bert",
    "gpt_neo",
    "gptj",
    "opt",
    "gpt-neox",
    "bloom",
}

SUPPORTED_TASKS = {
    "text-generation",
    "text-classification",
    "question-answering",
    "fill-mask",
    "token-classification",
    "conversational",
}

ARCHITECTURES_TO_TASK = {
    "ForCausalLM": "text-generation",
    "GPT2LMHeadModel": "text-generation",
    "ForSequenceClassification": "text-classification",
    "ForQuestionAnswering": "question-answering",
    "ForMaskedLM": "fill-mask",
    "ForTokenClassification": "token-classification",
}

TASK_TO_MODEL = {
    "text-generation": AutoModelForCausalLM,
    "text-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "fill-mask": AutoModelForMaskedLM,
    "token-classification": AutoModelForTokenClassification,
    "conversational": AutoModelForCausalLM,
}

MODEL_TYPE_TO_INJECTION_POLICY = {
    "roberta": {transformers.models.roberta.modeling_roberta.RobertaLayer: HFBertLayerPolicy},
    "gpt2": {transformers.models.gpt2.modeling_gpt2.GPT2Block: HFGPT2LayerPolicy},
    "bert": {transformers.models.bert.modeling_bert.BertLayer: HFBertLayerPolicy},
    "gpt_neo": {transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock: HFGPTNEOLayerPolicy},
    "gptj": {transformers.models.gptj.modeling_gptj.GPTJBlock: HFGPTJLayerPolicy},
    "opt": {transformers.models.opt.modeling_opt.OPTDecoderLayer: HFOPTLayerPolicy},
    "gpt-neox": {transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer: GPTNEOXLayerPolicy},
    "bloom": {transformers.models.bloom.modeling_bloom.BloomBlock: BLOOMLayerPolicy},
}


class DeepSpeedService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.task = None
        self.logger = logging.getLogger()
        self.model_dir = None
        self.model_id = None
        self.data_type = None
        self.max_tokens = None
        self.device = None
        self.world_size = None
        self.tensor_parallel_degree = None
        self.model_config = None

    def initialize(self, properties: dict):
        self.parse_properties(properties)
        self.validate_model_type_and_task()
        self.create_model_pipeline()
        self.logger.info(f"Initialized DeepSpeed model with the following configurations"
                         f"model: {self.model_id}"
                         f"task: {self.task}"
                         f"data_type: {self.data_type}"
                         f"tensor_parallel_degree: {self.tensor_parallel_degree}")
        self.initialized = True

    def parse_properties(self, properties):
        self.model_dir = properties.get("model_dir")
        self.model_id = properties.get("model_id")
        self.task = properties.get("task")
        self.data_type = properties.get("data_type", "fp32")
        self.max_tokens = int(properties.get("max_tokens", 1024))
        self.device = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.tensor_parallel_degree = int(properties.get("tensor_parallel_degree", self.world_size))
        self.ds_config = {
            "replace_with_kernel_inject": True,
            "dtype": self.data_type,
            "tensor_parallel": {
                "enabled": True,
                "tp_size": self.tensor_parallel_degree,
                "mpu": None,
            },
            "enable_cuda_graph": properties.get("enable_cuda_graph", "false").lower() == "true",
            "triangular_masking": properties.get("triangular_masking", "true").lower() == "true",
            "checkpoint": properties.get("checkpoint"),
            "base_dir": properties.get("base_dir"),
            "return_tuple": properties.get("return_tuple", "true").lower() == "true",
            "training_mp_size": int(properties.get("training_mp_size", 1)),
            "replace_method": "auto",
            "injection_policy": None,
            "max_out_tokens": self.max_tokens,
        }

    def validate_model_type_and_task(self):
        if not self.model_id:
            self.model_id = self.model_dir
            config_file = os.path.join(self.model_id, "config.json")
            if not os.path.exists(config_file):
                raise ValueError(f"model_dir: {self.model_id} does not contain a config.json. "
                                 f"This is required for loading models from local storage")
            self.model_config = AutoConfig.from_pretrained(config_file)
        else:
            self.model_config = AutoConfig.from_pretrained(self.model_id)

        if self.model_config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"model_type: {self.model_config.model_type} is not currently supported by DeepSpeed")

        if not self.task:
            self.logger.warning("No task provided. Attempting to infer from model architecture")
            self.infer_task_from_model_architecture(self.model_config)
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(f"task: {self.task} is not currently supported by DeepSpeed")

    def infer_task_from_model_architecture(self, config: PretrainedConfig):
        architecture = config.architectures[0]
        for arch_option in ARCHITECTURES_TO_TASK:
            if architecture.endswith(arch_option):
                self.task = ARCHITECTURES_TO_TASK[arch_option]

        if not self.task:
            raise ValueError(f"Task could not be inferred from model config. "
                             f"Please manually set `task` in serving.properties")

    def create_model_pipeline(self):
        # If a ds checkpoint is provided, we instantiate model with meta tensors. weights loaded when DS engine invoked
        if self.ds_config["checkpoint"]:
            dtype = torch.float32 if self.data_type == "fp32" else torch.float16
            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                model = TASK_TO_MODEL[self.task].from_config(self.model_config)
        else:
            model = TASK_TO_MODEL[self.task].from_pretrained(self.model_id, low_cpu_mem_usage=True)

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipeline = pipeline(task=self.task, model=model, tokenizer=tokenizer, device=self.device)
        self.logger.info(f"Model before deepspeed: {self.pipeline.model}")
        if self.model_config.model_type in MODEL_TYPE_TO_INJECTION_POLICY:
            self.ds_config["injection_policy"] = MODEL_TYPE_TO_INJECTION_POLICY[self.model_config.model_type]
        engine = deepspeed.init_inference(self.pipeline.model, **self.ds_config)
        self.pipeline.model = engine.module
        self.logger.info(f"Model after deepspeed: {self.pipeline.model}")

    def format_input_for_task(self, input_values):
        if not isinstance(input_values, list):
            input_values = [input_values]

        batch_inputs = []
        for val in input_values:
            if self.task == "conversational":
                current_input = Conversation(
                    text=val.get("text"),
                    conversation_id=val.get("conversation_id"),
                    past_user_inputs=val.get("past_user_inputs", []),
                    generated_responses=val.get("generated_responses", [])
                )
            elif self.task == "question-answering":
                current_input = SquadExample(
                    context_text=val.get("context"),
                    question_text=val.get("question")
                )
            else:
                current_input = val
            batch_inputs += [current_input]
        return batch_inputs

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            model_kwargs = {}
            if content_type is not None and content_type == "application/json":
                json_input = inputs.get_as_json()
                if isinstance(json_input, dict):
                    input_data = self.format_input_for_task(json_input.pop("inputs"))
                    model_kwargs = json_input
                else:
                    input_data = json_input
            else:
                input_data = inputs.get_as_string()

            result = self.pipeline(input_data, **model_kwargs)
            if self.task == "conversational":
                result = {
                    "generated_text": result.generated_responses[-1],
                    "conversation": {
                        "past_user_inputs": result.past_user_inputs,
                        "generated_responses": result.generated_responses,
                    },
                }

            outputs = Output()
            outputs.add(result)
        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            outputs = Output().error((str(e)))
        return outputs


_service = DeepSpeedService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)