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
import re

import torch
from transformers import (
    pipeline, Pipeline, Conversation, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, StoppingCriteria, StoppingCriteriaList)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import PeftConfig, PeftModel, PeftModelForCausalLM

from djl_python.encode_decode import encode
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.streaming_utils import StreamingUtils

from djl_python.properties_manager.properties import StreamingEnum, is_rolling_batch_enabled, is_streaming_enabled
from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.utils import parse_input_with_formatter, InputFormatConfigs

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
    "RWForCausalLM",
    "GPTNeoXForCausalLM",
    "T5ForConditionalGeneration",
    "LlamaForCausalLM",
    "FalconForCausalLM",
    "MPTForCausalLM",
    "GPTBigCodeForCausalLM",
}

# https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#efficient-inference-on-a-single-gpu
FLASH_2_SUPPORTED_MODELS = {
    "LlamaForCausalLM", "RWForCausalLM", "FalconForCausalLM"
}

PEFT_MODEL_TASK_TO_CLS = {
    "SEQ_CLS": AutoModelForSequenceClassification,
    "SEQ_2_SEQ_LM": AutoModelForSeq2SeqLM,
    "CAUSAL_LM": AutoModelForCausalLM,
    "TOKEN_CLS": AutoModelForTokenClassification,
    "QUESTION_ANS": AutoModelForQuestionAnswering,
}


def enable_flash():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
    return False


def get_rolling_batch_class_from_str(rolling_batch_type: str, is_mpi: bool,
                                     model_config):
    if rolling_batch_type == "auto":
        architecture = model_config.architectures[0]
        if architecture in LMI_DIST_ADV_MODEL and is_mpi:
            from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
            return LmiDistRollingBatch
        else:
            from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
            return SchedulerRollingBatch
    elif rolling_batch_type == "scheduler":
        from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
        return SchedulerRollingBatch
    elif rolling_batch_type == "lmi-dist":
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        return LmiDistRollingBatch
    elif rolling_batch_type == "vllm":
        from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
        return VLLMRollingBatch
    raise ValueError(f"Invalid rolling batch type: {rolling_batch_type}")


class StopWord(StoppingCriteria):

    def __init__(self, tokenizer, stop_seq):
        StoppingCriteria.__init__(self)
        self.tokenizer = tokenizer
        self.stop_seq = stop_seq

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs):
        decoded_input_ids = self.tokenizer.decode(
            input_ids[0][-len(self.stop_seq):])

        matches = re.search(self.stop_seq, decoded_input_ids)

        if matches is not None:
            return True

        return False


class HuggingFaceService(object):

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
        self.adapters = None
        self.hf_configs = None
        self.input_format_configs = None

    def initialize(self, properties: dict):
        self.hf_configs = HuggingFaceProperties(**properties)
        self._read_model_config(self.hf_configs.model_id_or_path)

        if is_rolling_batch_enabled(self.hf_configs.rolling_batch):
            _rolling_batch_cls = get_rolling_batch_class_from_str(
                self.hf_configs.rolling_batch.value, self.hf_configs.mpi_mode,
                self.model_config)
            self.hf_configs.kwargs["model_config"] = self.model_config
            self.rolling_batch = _rolling_batch_cls(
                self.hf_configs.model_id_or_path, properties,
                **self.hf_configs.kwargs)
            self._init_tokenizer(self.hf_configs.model_id_or_path)
        elif is_streaming_enabled(self.hf_configs.enable_streaming):
            self._init_tokenizer(self.hf_configs.model_id_or_path)
            self._init_model(self.hf_configs.model_id_or_path,
                             **self.hf_configs.kwargs)
        else:
            if not self.hf_configs.task:
                self.hf_configs.task = self.infer_task_from_model_architecture(
                )

            self.hf_pipeline = self.get_pipeline(
                task=self.hf_configs.task,
                model_id_or_path=self.hf_configs.model_id_or_path,
                kwargs=self.hf_configs.kwargs)

            if "stop_sequence" in properties:
                self.load_stopping_criteria_list(properties["stop_sequence"])

        self.input_format_configs = InputFormatConfigs(
            is_rolling_batch=is_rolling_batch_enabled(
                self.hf_configs.rolling_batch),
            is_adapters_supported=True,
            output_formatter=self.hf_configs.output_formatter,
            tokenizer=self.tokenizer)
        self.initialized = True

    @staticmethod
    def parse_stop_sequence_input(stop_sequence):
        """
        Gets a list of stop sequences by parsing the string given in
        serving.properties.
        Not robust against badly formatted input and commas in the stop sequence
        Input: stop_sequence (string)
        Output: list of strings
        """
        assert stop_sequence[0] == '[' and stop_sequence[
            -1] == ']', "option.stop_sequence not properly formatted"
        stop_sequence = stop_sequence.replace(", ", ",")
        stop_seq_list = [
            element[1:-1] for element in stop_sequence[1:-1].split(",")
        ]
        return stop_seq_list

    def load_stopping_criteria_list(self, stop_sequence):
        """
        Uses current tokenizer in self.tokenizer to load StoppingCriteriaList.
        Input: (str) stop_sequence - currently just one stop sequence supported
        Output: none (loads into member variable)
        """
        if self.tokenizer is None:
            return

        stop_seq_list = self.parse_stop_sequence_input(stop_sequence)

        stopwords = []
        for stop_seq in stop_seq_list:
            stopwords.append(StopWord(self.tokenizer, stop_seq))

        self.stopping_criteria_list = StoppingCriteriaList(stopwords)

    def parse_input(
        self, inputs: Input, tokenizer, output_formatter
    ) -> tuple[list[str], list[int], list[dict], dict, list]:
        """
        Preprocessing function that extracts information from Input objects.

        :param output_formatter: output formatter for the request
        :param inputs :(Input) a batch of inputs, each corresponding to a new request
        :param tokenizer: the tokenizer used for inference

        :return input_data (list[str]): a list of strings, each string being the prompt in a new request
        :return input_size (list[int]): a list of ints being the size of each new request
        :return parameters (list[dict]): parameters pertaining to each request
        :return errors (dict): a dictionary mapping int indices to corresponding error strings if any
        :return batch (list): a list of Input objects contained in inputs (each one corresponds to a request)
        """

        parsed_input = parse_input_with_formatter(inputs,
                                                  self.input_format_configs,
                                                  self.adapter_registry)
        self.adapters = parsed_input.adapters if parsed_input.found_adapters else None
        return parsed_input.input_data, parsed_input.input_size, parsed_input.parameters, parsed_input.errors, parsed_input.batch

    def inference(self, inputs):
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs, self.tokenizer, self.hf_configs.output_formatter)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                if is_rolling_batch_enabled(self.hf_configs.rolling_batch):
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                else:
                    outputs.add(err, key="data", batch_index=i)
            return outputs

        if is_rolling_batch_enabled(self.hf_configs.rolling_batch):
            if inputs.get_property("reset_rollingbatch"):
                self.rolling_batch.reset()
            if self.adapters is not None:
                adapter_data = [
                    self.adapter_registry.get(adapter, None)
                    for adapter in self.adapters
                ]
            else:
                adapter_data = None
            result = self.rolling_batch.inference(input_data,
                                                  parameters,
                                                  adapters=adapter_data)
            idx = 0
            for i in range(len(batch)):
                err = errors.get(i)
                if err:
                    err = {"data": "", "last": True, "code": 424, "error": err}
                    outputs.add(Output.binary_encode(err),
                                key="data",
                                batch_index=i)
                    outputs.add_property(f"batch_{i}_Content-Type",
                                         "application/json")
                else:
                    content_type = result[idx].pop("content_type")
                    outputs.add(Output.binary_encode(result[idx]),
                                key="data",
                                batch_index=i)
                    if content_type is not None:
                        outputs.add_property(f"batch_{i}_Content-Type",
                                             content_type)
                    idx += 1

            return outputs
        elif is_streaming_enabled(self.hf_configs.enable_streaming):
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.hf_configs.enable_streaming.value == StreamingEnum.huggingface.value:
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data,
                        self.hf_configs.device, **parameters[0]))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "Accelerate")
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     self.hf_configs.device, **parameters[0]))
            return outputs

        if not all(p == parameters[0] for p in parameters):
            raise ValueError(
                "In order to enable dynamic batching, all input batches must have the same parameters"
            )

        if isinstance(self.model, PeftModelForCausalLM):
            if self.adapters is None:
                # Inference with only base model
                self.adapters = [""] * len(input_data)
            parameters[0]["adapters"] = self.adapters

        prediction = self.hf_pipeline(input_data, **parameters[0])

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

        use_pipeline = True
        for element in ["load_in_8bit", "load_in_4bit", "low_cpu_mem_usage"]:
            if element in kwargs:
                use_pipeline = False
        # build pipeline
        if use_pipeline:
            if self.peft_config is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.peft_config.base_model_name_or_path,
                    trust_remote_code=self.hf_configs.trust_remote_code,
                    revision=self.hf_configs.revision,
                )
                base_model = PEFT_MODEL_TASK_TO_CLS[
                    self.peft_config.task_type].from_pretrained(
                        self.peft_config.base_model_name_or_path, **kwargs)
                self.model = PeftModel.from_pretrained(base_model,
                                                       model_id_or_path)
                if "load_in_8bit" in kwargs or "load_in_4bit" in kwargs:
                    logging.warning(
                        "LoRA checkpoints cannot be merged with base model when using 8bit or 4bit quantization."
                        "You should expect slightly longer inference times with a quantized model."
                    )
                else:
                    self.model = self.model.merge_and_unload()
                hf_pipeline = pipeline(task=task,
                                       tokenizer=self.tokenizer,
                                       model=self.model,
                                       device=self.hf_configs.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id_or_path,
                    revision=self.hf_configs.revision,
                    trust_remote_code=self.hf_configs.trust_remote_code)
                hf_pipeline = pipeline(task=task,
                                       tokenizer=self.tokenizer,
                                       model=model_id_or_path,
                                       device=self.hf_configs.device,
                                       **kwargs)
                self.model = hf_pipeline.model
        else:
            self._init_tokenizer(model_id_or_path)
            self._init_model(model_id_or_path, **kwargs)
            hf_pipeline = pipeline(task=task,
                                   model=self.model,
                                   tokenizer=self.tokenizer,
                                   device=self.hf_configs.device)

        # wrap specific pipeline to support better ux
        if task == "conversational":
            self.hf_pipeline_unwrapped = hf_pipeline
            hf_pipeline = self.wrap_conversation_pipeline(
                self.hf_pipeline_unwrapped)

        if task == "text-generation":
            if issubclass(type(hf_pipeline.tokenizer),
                          PreTrainedTokenizerBase):
                hf_pipeline.tokenizer.padding_side = "left"
                if not hf_pipeline.tokenizer.pad_token:
                    hf_pipeline.tokenizer.pad_token = hf_pipeline.tokenizer.eos_token
            self.hf_pipeline_unwrapped = hf_pipeline
            hf_pipeline = self.wrap_text_generation_pipeline(
                self.hf_pipeline_unwrapped)

        return hf_pipeline

    def _init_tokenizer(self, model_id_or_path: str):
        path_to_use = model_id_or_path if self.peft_config is None else self.peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            path_to_use,
            padding_size="left",
            trust_remote_code=self.hf_configs.trust_remote_code,
            revision=self.hf_configs.revision,
        )

    def _init_model(self, model_id_or_path: str, **kwargs):
        architectures = self.model_config.architectures
        if architectures and architectures[0].endswith(
                "ForConditionalGeneration"):
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM
            if architectures[0] in FLASH_2_SUPPORTED_MODELS and enable_flash(
            ) and not self.hf_configs.disable_flash_attn:
                kwargs['use_flash_attention_2'] = True

        if self.peft_config is not None:
            base_model = model_cls.from_pretrained(
                self.peft_config.base_model_name_or_path, **kwargs)
            self.model = PeftModel.from_pretrained(base_model,
                                                   model_id_or_path)
            if "load_in_8bit" in kwargs or "load_in_4bit" in kwargs:
                logging.warning(
                    "LoRA checkpoints cannot be merged with base model when using 8bit or 4bit quantization."
                    "You should expect slightly longer inference times with a quantized model."
                )
            else:
                self.model = self.model.merge_and_unload()
        else:
            self.model = model_cls.from_pretrained(model_id_or_path, **kwargs)

        if self.hf_configs.device:
            self.model.to(self.hf_configs.device)

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
            if self.hf_configs.device:
                input_tokens = input_tokens.to(self.hf_configs.device)
            else:
                input_tokens = input_tokens.to(model.device)

            with torch.no_grad():
                output_tokens = model.generate(
                    *args,
                    input_ids=input_tokens.input_ids,
                    attention_mask=input_tokens.attention_mask,
                    stopping_criteria=self.stopping_criteria_list,
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
                f"This is required for loading huggingface models")
            raise e


    def quantize(self, properties: dict):
        """
        Quantizes model using AutoAWQ. Saves output to save_mp_checkpoint_path.
        """
        from awq import AutoAWQForCausalLM

        logging.info("Preparing to quantize HuggingFace model")
        self.hf_configs = HuggingFaceProperties(**properties)
        self._read_model_config(self.hf_configs.model_id_or_path)
        self._init_tokenizer(self.hf_configs.model_id_or_path)
        logging.info(f"Initialized handler: {self}")

        output_path = self.hf_configs.save_mp_checkpoint_path
        # Hard-coding these options for now. If vLLM continues to prioritize 
        # AutoAWQ we will expose these options to customers in the future.
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        logging.info(f"Model loading kwargs: {self.hf_configs.kwargs}")
        awq_model = AutoAWQForCausalLM.from_pretrained(self.hf_configs.model_id_or_path, **self.hf_configs.kwargs)
        awq_model.quantize(self.tokenizer, quant_config=quant_config)

        logging.info(f"Saving model and tokenizer to: {output_path}")
        awq_model.save_quantized(output_path)
        self.tokenizer.save_pretrained(output_path)


_service = HuggingFaceService()


def register_adapter(inputs: Input):
    """
    Registers lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_path = inputs.get_property("src")
    if not os.path.exists(adapter_path):
        raise ValueError(
            f"Only local LoRA models are supported. {adapter_path} is not a valid path"
        )
    logging.info(f"Registering adapter {adapter_name} from {adapter_path}")
    _service.adapter_registry[adapter_name] = inputs
    if not is_rolling_batch_enabled(_service.hf_configs.rolling_batch):
        if isinstance(_service.model, PeftModel):
            _service.model.load_adapter(adapter_path, adapter_name)
        else:
            _service.model = PeftModel.from_pretrained(_service.model,
                                                       adapter_path,
                                                       adapter_name)

        if isinstance(_service.hf_pipeline, Pipeline):
            _service.hf_pipeline.model = _service.model

        if isinstance(_service.hf_pipeline_unwrapped, Pipeline):
            _service.hf_pipeline_unwrapped.model = _service.model
    return Output()


def unregister_adapter(inputs: Input):
    """
    Unregisters lora adapter from the model.
    """
    adapter_name = inputs.get_property("name")
    logging.info(f"Unregistering adapter {adapter_name}")
    #TODO: delete in vllm engine as well
    del _service.adapter_registry[adapter_name]
    if not is_rolling_batch_enabled(_service.hf_configs.rolling_batch):
        _service.model.base_model.delete_adapter(adapter_name)
    return Output()


def quantize(inputs: Input):
    if not _service.initialized:
        _service.quantize(inputs.get_properties())


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
