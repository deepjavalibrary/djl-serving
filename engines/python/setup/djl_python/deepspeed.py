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

# smoothquant reference -
# @InProceedings{xiao2023smoothquant,
#     title = {{S}mooth{Q}uant: Accurate and Efficient Post-Training Quantization for Large Language Models},
#     author = {Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
#     booktitle = {Proceedings of the 40th International Conference on Machine Learning},
#     year = {2023}
# }

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

from djl_python.encode_decode import decode, encode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.streaming_utils import StreamingUtils
from typing import Optional
from peft import PeftConfig, PeftModel

SUPPORTED_QUANTIZATION_MODE = [
    "smoothquant",
    "dynamic_int8"
]

SMOOTHQUANT_SUPPORTED_MODEL_TYPES = {
    "gpt2",
    "gpt_neo",
    "gptj",
    "gpt_neox",
    "bloom",
    "llama",
}

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
    "llama",
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
        self.peft_config = None
        self.model = None
        self.tokenizer = None
        self.revision = None

        self.quantize_mode = None
        self.smoothquant_alpha = None

    def initialize(self, properties: dict):
        self._parse_properties(properties)
        self._read_model_config()
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
        if "revision" in properties:
            self.revision = properties['revision']

        # SmoothQuant properties
        self._parse_smoothquant_properties(properties)

        if properties.get("deepspeed_config_path"):
            with open(properties.get("deepspeed_config_path"), "r") as f:
                self.ds_config = json.load(f)
        else:
            self.ds_config = self._get_ds_config(properties)

    def _parse_smoothquant_properties(self, properties):
        if 'quantize' in properties:
            if properties['quantize'] not in SUPPORTED_QUANTIZATION_MODE:
                self.logger.warning(
                    f"DeepSpeed does not currently support quantization mode: ${properties['quantize']}, "
                    f"this setting will be ignored.")
                return
            self.quantize_mode = properties['quantize']
        if 'smoothquant_alpha' in properties:
            try:
                if 0 <= float(properties['smoothquant_alpha']) <= 1:
                    self.smoothquant_alpha = properties['smoothquant_alpha']
                else:
                    raise ValueError(f"${properties['smoothquant_alpha']} is not between 0 and 1.")
            except ValueError:
                raise ValueError(f"${properties['smoothquant_alpha']} cannot convert to float number.")

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

        if self.quantize_mode:
            ds_config['dynamic_quant'] = {'enabled': True, 'use_cutlass': False}
            if self.quantize_mode == 'smoothquant':
                smoothing_value = {
                    'smooth': True,
                    'calibrate': True
                }
                if self.smoothquant_alpha:
                    smoothing_value['alpha'] = self.smoothquant_alpha
                ds_config['smoothing'] = smoothing_value
        return ds_config

    def _validate_model_type_and_task(self):
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

        if self.quantize_mode == \
                'smoothquant' and self.model_config.model_type not in SMOOTHQUANT_SUPPORTED_MODEL_TYPES:
            raise ValueError(f"${self.quantize_mode} does not support model ${self.model_config.model_type}")

    def _read_model_config(self):
        try:
            self.model_config = AutoConfig.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=self.trust_remote_code)
        except OSError:
            self.logger.warning(
                f"config.json not found for {self.model_id_or_path}. Attempting to load with peft"
            )
            self.peft_config = PeftConfig.from_pretrained(
                self.model_id_or_path)
            self.model_config = AutoConfig.from_pretrained(
                self.peft_config.base_model_name_or_path,
                trust_remote_code=self.trust_remote_code)
        except Exception as e:
            self.logger.error(
                f"{self.model_id_or_path} does not contain a config.json or adapter_config.json for lora models. "
                f"This is required for loading huggingface models")
            raise e

    def infer_task_from_model_architecture(self, config: PretrainedConfig):
        architecture = config.architectures[0]
        for arch_option in ARCHITECTURES_TO_TASK:
            if architecture.endswith(arch_option):
                self.task = ARCHITECTURES_TO_TASK[arch_option]

        if not self.task:
            raise ValueError(
                f"Task could not be inferred from model config. "
                f"Please manually set `task` in serving.properties.")

    def get_model_pretrained(self, model_id_or_path, torch_dtype='auto', **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = TASK_TO_MODEL[self.task].from_pretrained(model_id_or_path, torch_dtype=torch_dtype, **kwargs)
        return model, tokenizer

    def get_model_from_config(self, model_id_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = TASK_TO_MODEL[self.task].from_config(self.model_config, **kwargs)
        return model, tokenizer

    def get_model(self, model_id_or_path, loading_method, **kwargs):
        if loading_method == 'from_config':
            return self.get_model_from_config(model_id_or_path, **kwargs)
        elif loading_method == 'pretrained':
            return self.get_model_pretrained(model_id_or_path, **kwargs)
        else:
            raise RuntimeError(f'Unsupported model loading method, this should not happen.')

    def load_model(self, model_id_or_path, loading_method, use_mmap_loader, **kwargs):
        def load_model_with_mmap(model_id_or_path, loading_method):
            import mmaploader
            import accelerate
            with mmaploader.load_mmap_meta() as mmap_loader:
                with accelerate.init_empty_weights():
                    kwargs['low_cpu_mem_usage'] = False
                    model, tokenizer = self.get_model(model_id_or_path, loading_method, **kwargs)
            return model, tokenizer, mmap_loader.state_dict_mmap

        state_dict_mmap = {}
        model = None
        tokenizer = None
        done = False

        if use_mmap_loader:
            try:
                model, tokenizer, state_dict_mmap = load_model_with_mmap(model_id_or_path, loading_method)
                done = True
            except:
                self.logger.warning(f'failed to load model with mmap loader, will load model normally')

        if not done:
            kwargs['low_cpu_mem_usage'] = True
            model, tokenizer = self.get_model(model_id_or_path, loading_method, **kwargs)

        return model, tokenizer, state_dict_mmap

    def create_model_pipeline(self):
        # If a ds checkpoint is provided, we instantiate model with meta tensors. weights loaded when DS engine invoked
        # Workaround on int8. fp16 fp32 bf16 init supported
        dtype = torch.float16 if self.data_type == torch.int8 else self.data_type
        kwargs = {"torch_dtype": dtype} if dtype else {}
        if self.revision:
            kwargs['revision'] = self.revision
        if self.model_config.model_type in OPTIMIZED_MODEL_TYPES:
            self.ds_config["replace_with_kernel_inject"] = True
        else:
            self.ds_config["replace_with_kernel_inject"] = False
        state_dict_mmap = {}
        if "checkpoint" in self.ds_config:
            if self.quantize_mode:
                raise ValueError(
                    f"quantize option does NOT currently work WITH DeepSpeed checkpoints using checkpoint option. "
                    f"Please using quantization with a standard HuggingFace checkpoint or "
                    f"turn off quantization and try again.")
            model, self.tokenizer, state_dict_mmap = self.load_model(
                self.model_id_or_path,
                'from_config',
                self.ds_config['replace_with_kernel_inject'],
                **kwargs
            )
        elif self.peft_config is not None:
            self.logger.info(
                f"Peft Model detected. Instantiating base model {self.peft_config.base_model_name_or_path}"
            )
            base_model = TASK_TO_MODEL[self.task].from_pretrained(
                self.peft_config.base_model_name_or_path,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                **kwargs)
            lora_model = PeftModel.from_pretrained(base_model,
                                                   self.model_id_or_path)
            model = lora_model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(self.peft_config.base_model_name_or_path)
            self.logger.info(
                f"Peft Model merged into base model for deepspeed compatibility"
            )
        else:
            model, self.tokenizer, state_dict_mmap = self.load_model(
                self.model_id_or_path,
                'pretrained',
                self.ds_config['replace_with_kernel_inject'],
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
        if self.data_type:
            self.ds_config["dtype"] = self.data_type
        else:
            self.ds_config["dtype"] = model.dtype
        self.ds_config['replace_state_dict'] = state_dict_mmap

        # Optimization for text-generation batch processing
        if self.task == "text-generation":
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # If doing smoothquant calibration, set tokenizer
        smoothing_config = self.ds_config.get("smoothing", {})
        if smoothing_config.get("calibrate", False):
            smoothing_config["tokenizer"] = self.tokenizer

        self.model = deepspeed.init_inference(model, self.ds_config)

        # Don't create a "pipeline" if we're streaming or text-generation task, since those don't use a pipeline
        if self.enable_streaming or self.task == "text-generation":
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

    def parse_input(self, inputs):
        input_data = []
        input_size = []
        parameters = []
        errors = {}
        batch = inputs.get_batches()
        first = True
        for i, item in enumerate(batch):
            try:
                content_type = item.get_property("Content-Type")
                input_map = decode(item, content_type)
                _inputs = input_map.pop("inputs", input_map)
                if first:
                    parameters.append(input_map.pop("parameters", {}))
                    first = False
                else:
                    param = input_map.pop("parameters", {})
                    if parameters[0] != param:
                        logging.warning(
                            f"expected param: {parameters}, actual: {param}")
                        raise ValueError(
                            "In order to enable dynamic batching, all input batches must have the same parameters"
                        )
                if isinstance(_inputs, list):
                    input_data.extend(_inputs)
                    input_size.append(len(_inputs))
                else:
                    input_data.append(_inputs)
                    input_size.append(1)
            except Exception as e:  # pylint: disable=broad-except
                logging.exception(f"Parse input failed: {i}")
                errors[i] = str(e)

        return input_data, input_size, parameters, errors, batch

    def inference(self, inputs: Input):

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs)
        parameters = parameters[0]

        outputs = Output()
        if self.enable_streaming:
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.enable_streaming == "huggingface":
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data, self.device,
                        **parameters))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "DeepSpeed")
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     self.device, **parameters))
            return outputs
        if self.task == "text-generation":
            tokenized_inputs = self.tokenizer(input_data,
                                              padding=True,
                                              return_tensors="pt").to(
                                                  self.device)
            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    **parameters)
            prediction = self.tokenizer.batch_decode(output_tokens,
                                                     skip_special_tokens=True)
            prediction = [{"generated_text": s} for s in prediction]
        else:
            prediction = self.pipeline(input_data, **parameters)

        offset = 0
        for i, item in enumerate(batch):
            content_type = item.get_property("Content-Type")
            accept = item.get_property("Accept")
            if not accept:
                content_type = content_type if content_type else "application/json"
                accept = content_type if content_type.startswith(
                    "tensor/") else "application/json"
            elif "*/*" in accept:
                accept = "application/json"

            err = errors.get(i)
            if err:
                encode(outputs,
                       err,
                       accept,
                       key=inputs.get_content().key_at(i))
            else:
                encode(outputs,
                       prediction[offset:offset + input_size[i]],
                       accept,
                       key=inputs.get_content().key_at(i))
                offset += input_size[i]

        outputs.add_property("content-type", "application/json")

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
