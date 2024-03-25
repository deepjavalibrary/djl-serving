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
from djl_python.rolling_batch.rolling_batch import get_content_type_from_output_formatter

from djl_python.properties_manager.ds_properties import DeepSpeedProperties, DsQuantizeMethods
from djl_python.properties_manager.properties import StreamingEnum, is_streaming_enabled, is_rolling_batch_enabled

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


class DeepSpeedService(object):

    def __init__(self):
        self.properties = None
        self.pipeline = None
        self.initialized = False
        self.logger = logging.getLogger()
        self.model_config = None
        self.peft_config = None
        self.model = None
        self.tokenizer = None
        self.rolling_batch = None
        self.enable_rolling_batch = False

    def initialize(self, properties: dict):
        self.properties = DeepSpeedProperties(**properties)
        self.enable_rolling_batch = is_rolling_batch_enabled(
            self.properties.rolling_batch)
        self._read_model_config()
        self._validate_model_type_and_task()
        if self.enable_rolling_batch:
            from djl_python.rolling_batch.deepspeed_rolling_batch import DeepSpeedRollingBatch
            self.model = self.create_ds_module()
            if not self.properties.ds_config.get("replace_with_kernel_inject",
                                                 False):
                raise ValueError(
                    f"option.rolling_batch=deepspeed only works with kernel_injection models: {OPTIMIZED_MODEL_TYPES}"
                )
            kwargs = {
                "max_batch_size":
                int(properties.get("max_rolling_batch_size", 4)),
                "max_seq_len": int(properties.get("max_tokens", 1024)),
                "tokenizer": self.tokenizer
            }
            self.rolling_batch = DeepSpeedRollingBatch(self.model, properties,
                                                       **kwargs)
        else:
            self.create_model_pipeline()
        self.logger.info(
            f"Initialized DeepSpeed model with the following configurations\n"
            f"model: {self.properties.model_id_or_path}\n"
            f"task: {self.properties.task}\n"
            f"data_type: {self.properties.ds_config['dtype']}\n"
            f"tensor_parallel_degree: {self.properties.tensor_parallel_degree}\n"
            f"rolling_batch: {self.enable_rolling_batch}\n")
        self.initialized = True

    def _validate_model_type_and_task(self):
        if self.model_config.model_type not in OPTIMIZED_MODEL_TYPES:
            self.logger.warning(
                f"DeepSpeed does not currently support optimized CUDA kernels for the model type "
                f"{self.model_config.model_type}, and may not support this model for inference. Please "
                f"check the DeepSpeed documentation to verify. Attempting to load model with DeepSpeed."
            )

        if not self.properties.task:
            self.logger.warning(
                "No task provided. Attempting to infer from model architecture"
            )
            self.infer_task_from_model_architecture(self.model_config)
        if self.properties.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"task: {self.properties.task} is not currently supported by DeepSpeed"
            )

        if self.properties.quantize and \
                self.properties.quantize.value == DsQuantizeMethods.smoothquant.value \
                and self.model_config.model_type not in SMOOTHQUANT_SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"${self.properties.quantize.value} does not support model ${self.model_config.model_type}"
            )

    def _read_model_config(self):
        try:
            self.model_config = AutoConfig.from_pretrained(
                self.properties.model_id_or_path,
                trust_remote_code=self.properties.trust_remote_code)
        except OSError:
            self.logger.warning(
                f"config.json not found for {self.properties.model_id_or_path}. Attempting to load with peft"
            )
            self.peft_config = PeftConfig.from_pretrained(
                self.properties.model_id_or_path)
            self.model_config = AutoConfig.from_pretrained(
                self.peft_config.base_model_name_or_path,
                trust_remote_code=self.properties.trust_remote_code)
        except Exception as e:
            self.logger.error(
                f"{self.properties.model_id_or_path} "
                f"does not contain a config.json or adapter_config.json for lora models. "
                f"This is required for loading HuggingFace models")
            raise e

    def infer_task_from_model_architecture(self, config: PretrainedConfig):
        architecture = config.architectures[0]
        for arch_option in ARCHITECTURES_TO_TASK:
            if architecture.endswith(arch_option):
                self.properties.task = ARCHITECTURES_TO_TASK[arch_option]

        if not self.properties.task:
            raise ValueError(
                f"Task could not be inferred from model config. "
                f"Please manually set `task` in serving.properties.")

    def get_model_pretrained(self,
                             model_id_or_path,
                             torch_dtype='auto',
                             **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = TASK_TO_MODEL[self.properties.task].from_pretrained(
            model_id_or_path, torch_dtype=torch_dtype, **kwargs)
        return model, tokenizer

    def get_model_from_config(self, model_id_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = TASK_TO_MODEL[self.properties.task].from_config(
            self.model_config, **kwargs)
        return model, tokenizer

    def get_model(self, model_id_or_path, loading_method, **kwargs):
        if loading_method == 'from_config':
            if 'low_cpu_mem_usage' in kwargs:
                kwargs.pop('low_cpu_mem_usage')
            return self.get_model_from_config(model_id_or_path, **kwargs)
        elif loading_method == 'pretrained':
            return self.get_model_pretrained(model_id_or_path, **kwargs)
        else:
            raise RuntimeError(
                f'Unsupported model loading method, this should not happen.')

    def load_model(self, model_id_or_path, loading_method, use_mmap_loader,
                   **kwargs):

        def load_model_with_mmap(model_id_or_path, loading_method):
            import mmaploader
            import accelerate
            with mmaploader.load_mmap_meta() as mmap_loader:
                with accelerate.init_empty_weights():
                    kwargs['low_cpu_mem_usage'] = False
                    model, tokenizer = self.get_model(model_id_or_path,
                                                      loading_method, **kwargs)
            return model, tokenizer, mmap_loader.state_dict_mmap

        state_dict_mmap = {}
        model = None
        tokenizer = None
        done = False

        if use_mmap_loader:
            try:
                model, tokenizer, state_dict_mmap = load_model_with_mmap(
                    model_id_or_path, loading_method)
                done = True
            except:
                self.logger.warning(
                    f'failed to load model with mmap loader, will load model normally'
                )

        if not done:
            kwargs['low_cpu_mem_usage'] = True
            model, tokenizer = self.get_model(model_id_or_path, loading_method,
                                              **kwargs)

        return model, tokenizer, state_dict_mmap

    def create_ds_module(self):
        # If a ds checkpoint is provided, we instantiate model with meta tensors. weights loaded when DS engine invoked
        # Workaround on int8. fp16 fp32 bf16 init supported
        dtype = torch.float16 if self.properties.dtype == torch.int8 else self.properties.dtype
        kwargs = {"torch_dtype": dtype} if dtype else {}
        if self.properties.revision:
            kwargs['revision'] = self.properties.revision
        if self.model_config.model_type in OPTIMIZED_MODEL_TYPES:
            self.properties.ds_config["replace_with_kernel_inject"] = True
        else:
            self.properties.ds_config["replace_with_kernel_inject"] = False
        state_dict_mmap = {}
        if "checkpoint" in self.properties.ds_config:
            if self.properties.quantize:
                raise ValueError(
                    f"quantize option does NOT currently work WITH DeepSpeed checkpoints using checkpoint option. "
                    f"Please using quantization with a standard HuggingFace checkpoint or "
                    f"turn off quantization and try again.")
            model, self.tokenizer, state_dict_mmap = self.load_model(
                self.properties.model_id_or_path, 'from_config',
                self.properties.ds_config['replace_with_kernel_inject'],
                **kwargs)
        elif self.peft_config is not None:
            self.logger.info(
                f"Peft Model detected. Instantiating base model {self.peft_config.base_model_name_or_path}"
            )
            base_model = TASK_TO_MODEL[self.properties.task].from_pretrained(
                self.peft_config.base_model_name_or_path,
                low_cpu_mem_usage=self.properties.low_cpu_mem_usage,
                trust_remote_code=self.properties.trust_remote_code,
                **kwargs)
            lora_model = PeftModel.from_pretrained(
                base_model, self.properties.model_id_or_path)
            model = lora_model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.peft_config.base_model_name_or_path)
            self.logger.info(
                f"Peft Model merged into base model for deepspeed compatibility"
            )
        else:
            model, self.tokenizer, state_dict_mmap = self.load_model(
                self.properties.model_id_or_path,
                'pretrained',
                self.properties.ds_config['replace_with_kernel_inject'],
                low_cpu_mem_usage=self.properties.low_cpu_mem_usage,
                trust_remote_code=self.properties.trust_remote_code,
                **kwargs)
        if self.properties.dtype:
            self.properties.ds_config["dtype"] = self.properties.dtype
        else:
            self.properties.ds_config["dtype"] = model.dtype
        self.properties.ds_config['replace_state_dict'] = state_dict_mmap

        # Optimization for text-generation batch processing
        if self.properties.task == "text-generation":
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # If doing smoothquant calibration, set tokenizer
        smoothing_config = self.properties.ds_config.get("smoothing", {})
        if smoothing_config.get("calibrate", False):
            smoothing_config["tokenizer"] = self.tokenizer

        return deepspeed.init_inference(model, self.properties.ds_config)

    def create_model_pipeline(self):
        self.model = self.create_ds_module()
        # Don't create a "pipeline" if we're streaming or text-generation task, since those don't use a pipeline
        if is_streaming_enabled(self.properties.enable_streaming
                                ) or self.properties.task == "text-generation":
            return

        self.pipeline = pipeline(task=self.properties.task,
                                 model=self.model.module,
                                 tokenizer=self.tokenizer,
                                 device=self.properties.device)

    def format_input_for_task(self, input_values):
        if not isinstance(input_values, list):
            input_values = [input_values]

        batch_inputs = []
        for val in input_values:
            if self.properties.task == "conversational":
                current_input = Conversation(
                    text=val.get("text"),
                    conversation_id=val.get("conversation_id"),
                    past_user_inputs=val.get("past_user_inputs", []),
                    generated_responses=val.get("generated_responses", []))
            elif self.properties.task == "question-answering":
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
                _param = input_map.pop("parameters", {})
                _param["stream"] = input_map.pop("stream", False)
                if not self.enable_rolling_batch:
                    if first:
                        parameters.append(_param)
                        first = False
                    else:
                        if parameters[0] != _param:
                            logging.warning(
                                f"expected param: {parameters}, actual: {_param}"
                            )
                            raise ValueError(
                                "In order to enable dynamic batching, all input batches must have the same parameters"
                            )

                if "seed" not in _param:
                    # set server provided seed if seed is not part of request
                    if item.contains_key("seed"):
                        _param["seed"] = item.get_as_string(key="seed")

                if "output_formatter" not in _param:
                    _param[
                        "output_formatter"] = self.properties.output_formatter

                if not isinstance(_inputs, list):
                    _inputs = [_inputs]
                input_data.extend(_inputs)
                input_size.append(len(_inputs))
                if self.enable_rolling_batch:
                    for _ in range(input_size[i]):
                        parameters.append(_param)

            except Exception as e:  # pylint: disable=broad-except
                logging.exception(f"Parse input failed: {i}")
                errors[i] = str(e)

        return input_data, input_size, parameters, errors, batch

    def inference(self, inputs: Input):
        outputs = Output()
        input_data, input_size, params, errors, batch = self.parse_input(
            inputs)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                if self.enable_rolling_batch:
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                else:
                    outputs.add(err, key="data", batch_index=i)
            return outputs
        parameters = params[0]

        if self.enable_rolling_batch:
            if inputs.get_property("reset_rollingbatch"):
                self.rolling_batch.reset()
            result = self.rolling_batch.inference(input_data, params)
            idx = 0
            for i in range(len(batch)):
                err = errors.get(i)
                if err:
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                else:
                    outputs.add(Output.binary_encode(result[idx]),
                                key="data",
                                batch_index=i)
                    idx += 1

                formatter = parameters.get("output_formatter")
                content_type = get_content_type_from_output_formatter(
                    formatter)
                if content_type is not None:
                    outputs.add_property(f"batch_{i}_Content-Type",
                                         content_type)

            return outputs
        if is_streaming_enabled(self.properties.enable_streaming):
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.properties.enable_streaming.value == StreamingEnum.huggingface.value:
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data,
                        self.properties.device, **parameters))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "DeepSpeed")
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     self.properties.device, **parameters))
            return outputs
        if self.properties.task == "text-generation":
            tokenized_inputs = self.tokenizer(input_data,
                                              padding=True,
                                              return_tensors="pt").to(
                                                  self.properties.device)
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
