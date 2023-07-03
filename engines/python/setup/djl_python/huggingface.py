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

import logging
import os

import torch
from transformers import (pipeline, Conversation, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification,
                          AutoModelForQuestionAnswering)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import PeftConfig, PeftModel

from djl_python.encode_decode import encode, decode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.streaming_utils import StreamingUtils
from djl_python.rolling_batch import SchedulerRollingBatch

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

LMI_DIST_ADV_MODEL = {
    "RWForCausalLM", "GPTNeoXForCausalLM", "T5ForConditionalGeneration",
    "LlamaForCausalLM"
}

PEFT_MODEL_TASK_TO_CLS = {
    "SEQ_CLS": AutoModelForSequenceClassification,
    "SEQ_2_SEQ_LM": AutoModelForSeq2SeqLM,
    "CAUSAL_LM": AutoModelForCausalLM,
    "TOKEN_CLS": AutoModelForTokenClassification,
    "QUESTION_ANS": AutoModelForQuestionAnswering,
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


def get_rolling_batch_class_from_str(rolling_batch_type: str, is_mpi: bool,
                                     model_config):
    if rolling_batch_type == "auto":
        architecture = model_config.architectures[0]
        if architecture in LMI_DIST_ADV_MODEL and is_mpi:
            from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
            return LmiDistRollingBatch
        else:
            return SchedulerRollingBatch
    elif rolling_batch_type == "scheduler":
        return SchedulerRollingBatch
    elif rolling_batch_type == "lmi-dist":
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        return LmiDistRollingBatch
    raise ValueError(f"Invalid rolling batch type: {rolling_batch_type}")


class HuggingFaceService(object):

    def __init__(self):
        self.hf_pipeline = None
        self.initialized = False
        self.enable_streaming = None
        self.model = None
        self.device = None
        self.tokenizer = None
        self.trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE",
                                                "FALSE").lower() == 'true'
        self.rolling_batch_type = None
        self.rolling_batch = None
        self.model_config = None
        self.peft_config = None

    def initialize(self, properties: dict):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        device_id = int(properties.get("device_id", "-1"))
        self.device = f"cuda:{device_id}" if device_id >= 0 else None
        task = properties.get("task")
        tp_degree = int(properties.get("tensor_parallel_degree", "-1"))
        self.enable_streaming = properties.get("enable_streaming", None)
        if self.enable_streaming and self.enable_streaming.lower() == "false":
            self.enable_streaming = None
        if "trust_remote_code" in properties:
            self.trust_remote_code = properties.get(
                "trust_remote_code").lower() == "true"
        # HF Acc handling
        kwargs = {"trust_remote_code": self.trust_remote_code}
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        if "device_map" in properties:
            kwargs["device_map"] = properties.get("device_map")
            self.device = None
            logging.info(f"Using device map {kwargs['device_map']}")
        elif tp_degree > 0 and torch.cuda.device_count() > 0:
            kwargs["device_map"] = "auto"
            self.device = None
            world_size = torch.cuda.device_count()
            assert world_size == tp_degree, f"TP degree ({tp_degree}) doesn't match available GPUs ({world_size})"
            logging.info(f"Using {world_size} gpus")
        if "load_in_8bit" in properties:
            if "device_map" not in kwargs:
                raise ValueError(
                    "device_map should set when load_in_8bit is set")
            kwargs["load_in_8bit"] = properties.get("load_in_8bit")
        if "load_in_4bit" in properties:
            if "device_map" not in kwargs:
                raise ValueError(
                    "device_map should set when load_in_4bit is set")
            kwargs["load_in_8bit"] = properties.get("load_in_4bit")
        if "low_cpu_mem_usage" in properties:
            kwargs["low_cpu_mem_usage"] = properties.get("low_cpu_mem_usage")

        if "data_type" in properties:
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties.get("data_type"))
        if "dtype" in properties:
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties.get("dtype"))
        self.rolling_batch_type = properties.get("rolling_batch", None)

        self._read_model_config(model_id_or_path)

        if self.rolling_batch_type:
            self.rolling_batch_type = self.rolling_batch_type.lower()
            is_mpi = properties.get("engine") != "Python"
            if is_mpi:
                self.device = int(os.getenv("LOCAL_RANK", 0))
            _rolling_batch_cls = get_rolling_batch_class_from_str(
                self.rolling_batch_type, is_mpi, self.model_config)
            self.rolling_batch = _rolling_batch_cls(model_id_or_path,
                                                    self.device, properties,
                                                    **kwargs)
            self.initialized = True
            return
        elif self.enable_streaming:
            self._init_model_and_tokenizer(model_id_or_path, **kwargs)
            self.initialized = True
            return

        if not task:
            task = self.infer_task_from_model_architecture()

        self.hf_pipeline = self.get_pipeline(task=task,
                                             model_id_or_path=model_id_or_path,
                                             kwargs=kwargs)

        self.initialized = True

    def inference(self, inputs):
        content_type = inputs.get_property("Content-Type")
        accept = inputs.get_property("Accept")
        if not accept:
            accept = content_type if content_type.startswith(
                "tensor/") else "application/json"
        elif "*/*" in accept:
            accept = "application/json"

        input_data = []
        input_size = []
        parameters = []
        batch = inputs.get_batches()
        first = True
        for item in batch:
            input_map = decode(item, content_type)
            _inputs = input_map.pop("inputs", input_map)
            if isinstance(_inputs, list):
                input_data.extend(_inputs)
                input_size.append(len(_inputs))
            else:
                input_data.append(_inputs)
                input_size.append(1)
            if first or self.rolling_batch_type:
                parameters.append(input_map.pop("parameters", {}))
                first = False
            else:
                if parameters != input_map.pop("parameters", {}):
                    return Output().error(
                        "In order to enable dynamic batching, all input batches must have the same parameters"
                    )

        outputs = Output()

        if self.rolling_batch_type:
            result = self.rolling_batch.inference(input_data, parameters)
            for i in range(inputs.get_batch_size()):
                res = result[i]
                encode(outputs,
                       res,
                       accept,
                       key=inputs.get_content().key_at(i))

            return outputs
        elif self.enable_streaming:
            outputs.add_property("content-type", "application/jsonlines")
            if self.enable_streaming == "huggingface":
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data, self.device,
                        **parameters[0]))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "Accelerate")
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     self.device, **parameters[0]))
            return outputs

        prediction = self.hf_pipeline(input_data, **parameters[0])

        offset = 0
        for i in range(inputs.get_batch_size()):
            encode(outputs,
                   prediction[offset:offset + input_size[i]],
                   accept,
                   key=inputs.get_content().key_at(i))
            offset += input_size[i]

        return outputs

    def get_pipeline(self, task: str, model_id_or_path: str, kwargs):
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
        for element in ["load_in_8bit", "load_in_4bit", "low_cpu_mem_usage"]:
            if element in kwargs:
                use_pipeline = False
        # build pipeline
        if use_pipeline:
            if self.peft_config is not None:
                kwargs.pop("tokenizer", None)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.peft_config.base_model_name_or_path)
                base_model = PEFT_MODEL_TASK_TO_CLS[
                    self.peft_config.task_type].from_pretrained(
                        self.peft_config.base_model_name_or_path, **kwargs)
                lora_model = PeftModel.from_pretrained(base_model,
                                                       model_id_or_path)
                self.model = lora_model.merge_and_unload()
                hf_pipeline = pipeline(task=task,
                                       tokenizer=self.tokenizer,
                                       model=self.model,
                                       device=self.device)
            else:
                hf_pipeline = pipeline(task=task,
                                       model=model_id_or_path,
                                       device=self.device,
                                       **kwargs)
        else:
            kwargs.pop("tokenizer", None)
            self._init_model_and_tokenizer(model_id_or_path, **kwargs)
            hf_pipeline = pipeline(task=task,
                                   model=self.model,
                                   tokenizer=self.tokenizer,
                                   device=self.device)

        # wrap specific pipeline to support better ux
        if task == "conversational":
            hf_pipeline = self.wrap_conversation_pipeline(hf_pipeline)

        if task == "text-generation":
            if issubclass(type(hf_pipeline.tokenizer),
                          PreTrainedTokenizerBase):
                hf_pipeline.tokenizer.padding_side = "left"
                if not hf_pipeline.tokenizer.pad_token:
                    hf_pipeline.tokenizer.pad_token = hf_pipeline.tokenizer.eos_token
            hf_pipeline = self.wrap_text_generation_pipeline(hf_pipeline)

        return hf_pipeline

    def _init_model_and_tokenizer(self, model_id_or_path: str, **kwargs):
        if self.peft_config is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.peft_config.base_model_name_or_path, padding_size="left")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                           padding_side="left")
        architectures = self.model_config.architectures
        if architectures and architectures[0].endswith(
                "ForConditionalGeneration"):
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        if self.peft_config is not None:
            base_model = model_cls.from_pretrained(
                self.peft_config.base_model_name_or_path, **kwargs)
            lora_model = PeftModel.from_pretrained(base_model,
                                                   model_id_or_path)
            self.model = lora_model.merge_and_unload()
        else:
            self.model = model_cls.from_pretrained(model_id_or_path, **kwargs)

        if self.device:
            self.model.to(self.device)

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

    def wrap_text_generation_pipeline(self, hf_pipeline):

        def wrapped_pipeline(inputs, *args, **kwargs):
            model = hf_pipeline.model
            tokenizer = hf_pipeline.tokenizer
            input_tokens = tokenizer(inputs, padding=True, return_tensors="pt")
            if self.device:
                input_tokens = input_tokens.to(self.device)

            with torch.no_grad():
                output_tokens = model.generate(
                    *args,
                    input_ids=input_tokens.input_ids,
                    attention_mask=input_tokens.attention_mask,
                    **kwargs)
            generated_text = tokenizer.batch_decode(output_tokens,
                                                    skip_special_tokens=True)

            return [{"generated_text": s} for s in generated_text]

        return wrapped_pipeline

    def infer_task_from_model_architecture(self):
        architecture = self.model_config.architectures[0]

        task = None
        for arch_options in ARCHITECTURES_2_TASK:
            if architecture.endswith(arch_options):
                task = ARCHITECTURES_2_TASK[arch_options]

        if task is None:
            raise ValueError(
                f"Task couldn't be inferred from {architecture}. Please manually set `task` option."
            )
        return task

    def _read_model_config(self, model_config_path: str):
        try:
            self.model_config = AutoConfig.from_pretrained(
                model_config_path, trust_remote_code=self.trust_remote_code)
        except OSError:
            logging.warning(
                f"config.json not found for {model_config_path}. Attempting to load with peft"
            )
            self.peft_config = PeftConfig.from_pretrained(model_config_path)
            self.model_config = AutoConfig.from_pretrained(
                self.peft_config.base_model_name_or_path)
        except Exception as e:
            self.logger.error(
                f"{self.model_id_or_path} does not contain a config.json or adapter_config.json for lora models. "
                f"This is required for loading huggingface models")
            raise e


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
