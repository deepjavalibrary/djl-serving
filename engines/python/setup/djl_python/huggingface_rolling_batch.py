#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from peft import PeftConfig
from transformers import AutoTokenizer, AutoConfig
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.utils import rolling_batch_inference
from djl_python.input_parser import parse_input_with_formatter


def get_rolling_batch_class_from_str(rolling_batch_type: str):
    if rolling_batch_type == "scheduler":
        from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
        return SchedulerRollingBatch
    elif rolling_batch_type == "lmi-dist":
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        return LmiDistRollingBatch
    elif rolling_batch_type == "vllm":
        from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
        return VLLMRollingBatch
    raise ValueError(f"Invalid rolling batch type: {rolling_batch_type}")


class HuggingFaceRollingBatchService(object):

    def __init__(self):
        self.hf_pipeline = None
        self.hf_pipeline_unwrapped = None
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.rolling_batch = None
        self.model_config = None
        self.peft_config = None
        self.stopping_criteria_list = None
        self.adapter_registry = {}
        self.hf_configs = None
        self.input_format_args = None

    def initialize(self, properties: dict):
        self.hf_configs = HuggingFaceProperties(**properties)
        self._read_model_config(self.hf_configs.model_id_or_path)

        _rolling_batch_cls = get_rolling_batch_class_from_str(
            self.hf_configs.rolling_batch.value)
        self.hf_configs.kwargs["model_config"] = self.model_config
        self.rolling_batch = _rolling_batch_cls(
            self.hf_configs.model_id_or_path, properties,
            **self.hf_configs.kwargs)
        self._init_tokenizer(self.hf_configs.model_id_or_path)

        self.input_format_args = self.get_input_format_args()
        self.initialized = True

    def get_input_format_args(self):
        return {
            "configs": self.hf_configs,
            "tokenizer": self.tokenizer,
            "adapter_registry": self.adapter_registry,
            "model_config": self.model_config,
            "peft_config": self.peft_config,
            "rolling_batch": self.rolling_batch,
            "image_placeholder_token": self.get_image_token(),
        }

    def inference(self, inputs: Input) -> Output:
        outputs = Output()
        parsed_input = parse_input_with_formatter(inputs,
                                                  **self.input_format_args)
        errors = parsed_input.errors
        if errors and len(parsed_input.batch) == len(errors):
            for i in range(len(parsed_input.batch)):
                err = errors.get(i)
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            return outputs

        return rolling_batch_inference(parsed_input, inputs, outputs,
                                       self.rolling_batch)

    def _init_tokenizer(self, model_id_or_path: str):
        path_to_use = model_id_or_path if self.peft_config is None else self.peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            path_to_use,
            padding_size="left",
            trust_remote_code=self.hf_configs.trust_remote_code,
            revision=self.hf_configs.revision,
        )

    def _read_model_config(self, model_config_path: str):
        try:
            self.model_config = AutoConfig.from_pretrained(
                model_config_path,
                trust_remote_code=self.hf_configs.trust_remote_code,
                revision=self.hf_configs.revision)
        except OSError:
            logging.warning(
                f"config.json not found for {model_config_path}. Attempting to load with peft"
            )
            self.peft_config = PeftConfig.from_pretrained(model_config_path)
            self.model_config = AutoConfig.from_pretrained(
                self.peft_config.base_model_name_or_path,
                trust_remote_code=self.hf_configs.trust_remote_code,
                revision=self.hf_configs.revision,
            )
        except Exception as e:
            logging.error(
                f"{model_config_path} does not contain a config.json or adapter_config.json for lora models. "
                f"This is required for loading huggingface models",
                exc_info=True)
            raise e

    def get_image_token(self):
        model_type = self.model_config.model_type
        if model_type == "phi3_v":
            return "<|image_{}|>"
        if model_type == "minicpmv":
            return "(<image>./</image>)"
        if model_type in ("blip-2", "chatglm", "fuyu", "paligemma", "pixtral"):
            # These models do not use image tokens in the prompt
            return None
        if model_type == "qwen":
            return "Picture {}: <img></img>"
        if model_type.startswith("llava"):
            return "<image>"
        if model_type in ("chameleon", "internvl_chat"):
            return "<image>"
        if model_type == "mllama":
            return "<|image|>"
        if model_type == "qwen2_vl":
            return "<|vision_start|><|image_pad|><|vision_end|>"

        logging.warning(
            "could not infer image token from the model artifacts. Using <image> as default."
        )
        return "<image>"
