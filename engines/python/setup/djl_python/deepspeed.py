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
import json
import torch
from transformers import (AutoConfig, PretrainedConfig, AutoTokenizer,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForQuestionAnswering, AutoModelForMaskedLM,
                          AutoModelForTokenClassification, pipeline,
                          Conversation, SquadExample)
import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.streaming_utils import StreamingUtils
from typing import Optional

OPTIMIZED_MODEL_TYPES = {
    "roberta",
    "xlm-roberta",
    "gpt2",
    "bert",
    "gpt_neo",
    "gptj",
    "opt",
    "gpt_neox",
    "bloom",
}

SUPPORTED_TASKS = {
    "text-generation",
    "text-classification",
    "question-answering",
    "fill-mask",
    "token-classification",
    "conversational",
    "text2text-generation",
}

ARCHITECTURES_TO_TASK = {
    "ForCausalLM": "text-generation",
    "GPT2LMHeadModel": "text-generation",
    "ForSequenceClassification": "text-classification",
    "ForQuestionAnswering": "question-answering",
    "ForMaskedLM": "fill-mask",
    "ForTokenClassification": "token-classification",
    "BloomModel": "text-generation",
    "ForConditionalGeneration": "text2text-generation",
}

TASK_TO_MODEL = {
    "text-generation": AutoModelForCausalLM,
    "text-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "fill-mask": AutoModelForMaskedLM,
    "token-classification": AutoModelForTokenClassification,
    "conversational": AutoModelForCausalLM,
    "text2text-generation": AutoModelForSeq2SeqLM
}


def get_torch_dtype_from_str(dtype: str):
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


def default_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


class DeepSpeedService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.task = None
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.max_tokens = None
        self.device = None
        self.tensor_parallel_degree = None
        self.model_config = None
        self.low_cpu_mem_usage = False
        self.enable_streaming = None
        self.trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE",
                                                "FALSE").lower() == 'true'
        self.model = None
        self.tokenizer = None

    def initialize(self, properties: dict):
        self._parse_properties(properties)
        self._validate_model_type_and_task()
        self.create_model_pipeline()
        self.logger.info(
            f"Initialized DeepSpeed model with the following configurations\n"
            f"model: {self.model_id_or_path}\n"
            f"task: {self.task}\n"
            f"data_type: {self.ds_config['dtype']}\n"
            f"tensor_parallel_degree: {self.tensor_parallel_degree}\n")
        self.initialized = True

    def _parse_properties(self, properties):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.task = properties.get("task")
        self.data_type = get_torch_dtype_from_str(
            properties.get("dtype", default_dtype()))
        self.max_tokens = int(properties.get("max_tokens", 1024))
        self.device = int(os.getenv("LOCAL_RANK", 0))
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        self.low_cpu_mem_usage = properties.get("low_cpu_mem_usage",
                                                "true").lower() == "true"
        self.enable_streaming = properties.get("enable_streaming", None)
        if self.enable_streaming and self.enable_streaming.lower() == "false":
            self.enable_streaming = None
        if "trust_remote_code" in properties:
            self.trust_remote_code = properties.get(
                "trust_remote_code").lower() == "true"
        if properties.get("deepspeed_config_path"):
            with open(properties.get("deepspeed_config_path"), "r") as f:
                self.ds_config = json.load(f)
        else:
            self.ds_config = self._get_ds_config(properties)

    def _get_ds_config(self, properties: dict):
        ds_config = {
            "tensor_parallel": {
                "tp_size": self.tensor_parallel_degree
            },
            "enable_cuda_graph":
            properties.get("enable_cuda_graph", "false").lower() == "true",
            "triangular_masking":
            properties.get("triangular_masking", "true").lower() == "true",
            "return_tuple":
            properties.get("return_tuple", "true").lower() == "true",
            "training_mp_size":
            int(properties.get("training_mp_size", 1)),
            "max_tokens":
            self.max_tokens,
            "save_mp_checkpoint_path":
            properties.get("save_mp_checkpoint_path")
        }
        if "checkpoint" in properties:
            ds_config["checkpoint"] = os.path.join(
                self.model_id_or_path, properties.get("checkpoint"))
            ds_config["base_dir"] = self.model_id_or_path
            if self.data_type is None:
                raise ValueError(
                    "dtype should also be provided for checkpoint loading")
        return ds_config

    def _validate_model_type_and_task(self):
        if os.path.exists(self.model_id_or_path):
            config_file = os.path.join(self.model_id_or_path, "config.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"{self.model_id_or_path} does not contain a config.json. "
                    f"This is required for loading models from local storage")
            self.model_config = AutoConfig.from_pretrained(
                config_file, trust_remote_code=self.trust_remote_code)
        else:
            self.model_config = AutoConfig.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=self.trust_remote_code)

        if self.model_config.model_type not in OPTIMIZED_MODEL_TYPES:
            self.logger.warning(
                f"DeepSpeed does not currently support optimized CUDA kernels for the model type "
                f"{self.model_config.model_type}, and may not support this model for inference. Please "
                f"check the DeepSpeed documentation to verify. Attempting to load model with DeepSpeed."
            )

        if not self.task:
            self.logger.warning(
                "No task provided. Attempting to infer from model architecture"
            )
            self.infer_task_from_model_architecture(self.model_config)
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"task: {self.task} is not currently supported by DeepSpeed")

    def infer_task_from_model_architecture(self, config: PretrainedConfig):
        architecture = config.architectures[0]
        for arch_option in ARCHITECTURES_TO_TASK:
            if architecture.endswith(arch_option):
                self.task = ARCHITECTURES_TO_TASK[arch_option]

        if not self.task:
            raise ValueError(
                f"Task could not be inferred from model config. "
                f"Please manually set `task` in serving.properties.")

    def create_model_pipeline(self):
        # If a ds checkpoint is provided, we instantiate model with meta tensors. weights loaded when DS engine invoked
        # Workaround on int8. fp16 fp32 bf16 init supported
        dtype = torch.float16 if self.data_type == torch.int8 else self.data_type
        kwargs = {"torch_dtype": dtype} if dtype else {}
        if "checkpoint" in self.ds_config:
            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                model = TASK_TO_MODEL[self.task].from_config(
                    self.model_config, **kwargs)
        else:
            model = TASK_TO_MODEL[self.task].from_pretrained(
                self.model_id_or_path,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                **kwargs)
        if self.data_type:
            self.ds_config["dtype"] = self.data_type
        else:
            self.ds_config["dtype"] = model.dtype
        if self.model_config.model_type in OPTIMIZED_MODEL_TYPES:
            self.ds_config["replace_with_kernel_inject"] = True
        self.model = deepspeed.init_inference(model, config=self.ds_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
        if self.enable_streaming:
            return
        # Optimization for text-generation batch processing
        if self.task == "text-generation":
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        self.pipeline = pipeline(task=self.task,
                                 model=self.model.module,
                                 tokenizer=self.tokenizer,
                                 device=self.device)

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
                    generated_responses=val.get("generated_responses", []))
            elif self.task == "question-answering":
                current_input = SquadExample(None, val.get("context"),
                                             val.get("question"), None, None,
                                             None)
            else:
                current_input = val
            batch_inputs += [current_input]
        return batch_inputs

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            model_kwargs = {}
            if content_type is not None and content_type.startswith(
                    "application/json"):
                json_input = inputs.get_as_json()
                if isinstance(json_input, dict):
                    input_data = self.format_input_for_task(
                        json_input.pop("inputs"))
                    model_kwargs = json_input.pop("parameters", {})
                else:
                    input_data = json_input
            else:
                input_data = inputs.get_as_string()

            outputs = Output()
            if self.enable_streaming:
                outputs.add_property("content-type", "application/jsonlines")
                if self.enable_streaming == "huggingface":
                    outputs.add_stream_content(
                        StreamingUtils.use_hf_default_streamer(
                            self.model, self.tokenizer, input_data, 0,
                            **model_kwargs))
                else:
                    stream_generator = StreamingUtils.get_stream_generator(
                        "DeepSpeed")
                    outputs.add_stream_content(
                        stream_generator(self.model, self.tokenizer,
                                         input_data, **model_kwargs))
                return outputs
            if self.task == "text-generation":
                tokenized_inputs = self.tokenizer(
                    input_data, padding=True,
                    return_tensors="pt").to(torch.cuda.current_device())
                with torch.no_grad():
                    output_tokens = self.model.generate(
                        input_ids=tokenized_inputs.input_ids,
                        attention_mask=tokenized_inputs.attention_mask,
                        **model_kwargs)
                generated_text = self.tokenizer.batch_decode(
                    output_tokens, skip_special_tokens=True)
                outputs.add([{"generated_text": s} for s in generated_text])
                outputs.add_property("content-type", "application/json")
                return outputs

            result = self.pipeline(input_data, **model_kwargs)
            if self.task == "conversational":
                result = {
                    "generated_text": result.generated_responses[-1],
                    "conversation": {
                        "past_user_inputs": result.past_user_inputs,
                        "generated_responses": result.generated_responses,
                    },
                }
                outputs.add_property("content-type", "application/json")

            outputs.add(result)
        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            outputs = Output().error((str(e)))
        return outputs


_service = DeepSpeedService()


def partition(inputs: Input):
    _service.initialize(inputs.get_properties())


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)
