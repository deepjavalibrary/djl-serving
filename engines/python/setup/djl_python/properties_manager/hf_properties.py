import logging
import os
from enum import Enum
from typing import Optional, Union

import torch

from pydantic import validator, root_validator

from djl_python.properties_manager.properties import Properties, RollingBatchEnum


class HFQuantizeMethods(str, Enum):
    # added for backward compatibility lmi-dist
    bitsandbytes = 'bitsandbytes'
    gptq = 'gptq'

    # huggingface
    bitsandbytes4 = 'bitsandbytes4'
    bitsandbytes8 = 'bitsandbytes8'

    # supported by vllm
    awq = 'awq'


HF_SUPPORTED_ROLLING_BATCH_TYPES = [
    RollingBatchEnum.scheduler.value, RollingBatchEnum.lmidist.value,
    RollingBatchEnum.vllm.value, RollingBatchEnum.auto.value,
    RollingBatchEnum.disable.value
]

LMI_DIST_ADV_MODEL = {
    "RWForCausalLM",
    "GPTNeoXForCausalLM",
    "T5ForConditionalGeneration",
    "LlamaForCausalLM",
    "FalconForCausalLM",
    "MPTForCausalLM",
    "GPTBigCodeForCausalLM",
}


class HuggingFaceProperties(Properties):
    device_id: int = -1
    task: str = None
    tensor_parallel_degree: int = -1
    device_map: str = None
    load_in_4bit: Optional[bool] = None
    load_in_8bit: Optional[bool] = None
    quantize: Optional[HFQuantizeMethods] = None
    low_cpu_mem_usage: Optional[bool] = False
    disable_flash_attn: Optional[bool] = True
    output_formatter: Optional[str]
    waiting_steps: Optional[int]

    device: Optional[str] = None
    is_mpi: bool = False
    kwargs: Optional[dict] = {}

    @root_validator(pre=True)
    def calculate_is_mpi(cls, properties):
        properties['is_mpi'] = properties.get("engine") != "Python"
        return properties

    @validator('rolling_batch', pre=True)
    def validate_rolling_batch(cls, rolling_batch: str) -> str:
        rolling_batch = rolling_batch.lower()
        if rolling_batch == RollingBatchEnum.disable.value:
            return rolling_batch
        if rolling_batch not in HF_SUPPORTED_ROLLING_BATCH_TYPES:
            logging.warning(
                f"huggingface handler only supports "
                f"rolling batch type {HF_SUPPORTED_ROLLING_BATCH_TYPES}."
                f"choosing auto mode for rolling batch automatically.")
            return 'auto'
        return rolling_batch

    @validator('load_in_4bit')
    def validate_load_in_4bit(cls, load_in_4bit):
        logging.warning('option.load_in_4bit is deprecated. '
                        'Kindly use option.quantize=bitsandbytes4 instead')
        return load_in_4bit

    @validator('load_in_8bit')
    def validate_load_in_8bit(cls, load_in_8bit):
        logging.warning('option.load_in_8bit is deprecated. '
                        'Kindly use option.quantize=bitsandbytes8 instead')
        return load_in_8bit

    @root_validator()
    def set_quantize_for_backward_compatibility(cls, properties):
        if properties['load_in_4bit']:
            properties['quantize'] = HFQuantizeMethods.bitsandbytes4
        elif properties['load_in_8bit']:
            properties['quantize'] = HFQuantizeMethods.bitsandbytes8

        # parsing bitsandbytes8, so it can be directly passed to lmi dist model loader.
        if properties['quantize'] == HFQuantizeMethods.bitsandbytes8 \
                and properties['rolling_batch'] == RollingBatchEnum.lmidist:
            properties['quantize'] = HFQuantizeMethods.bitsandbytes
        return properties

    @root_validator()
    def set_device(cls, properties):
        if properties['device_id'] >= 0:
            properties['device'] = f"cuda:{properties['device_id']}"
        else:
            properties['device'] = None
        return properties

    @root_validator()
    def construct_kwargs(cls, properties):
        kwargs = properties['kwargs']

        kwargs['trust_remote_code'] = properties['trust_remote_code']
        if properties['low_cpu_mem_usage']:
            kwargs["low_cpu_mem_usage"] = properties['low_cpu_mem_usage']
        if properties['revision']:
            kwargs["revision"] = properties['revision']

        if properties['rolling_batch'].value != RollingBatchEnum.disable.value:
            if properties['output_formatter']:
                kwargs["output_formatter"] = properties['output_formatter']
            if properties['waiting_steps']:
                kwargs["waiting_steps"] = properties['waiting_steps']

        properties['kwargs'] = kwargs
        return properties

    @root_validator()
    def construct_kwargs_device_map(cls, properties):
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        kwargs = properties['kwargs']
        if properties['device_map']:
            kwargs["device_map"] = properties['device_map']
            properties['device'] = None
            logging.info(f"Using device map {properties['device_map']}")
        elif properties[
                'tensor_parallel_degree'] > 0 and torch.cuda.device_count(
                ) > 0:
            kwargs["device_map"] = "auto"
            properties['device'] = None
            world_size = torch.cuda.device_count()
            assert world_size == properties['tensor_parallel_degree'], \
                f"TP degree ({properties['tensor_parallel_degree']}) doesn't match available GPUs ({world_size})"
            logging.info(f"Using {world_size} gpus")

        properties['kwargs'] = kwargs
        return properties

    @root_validator()
    def construct_kwargs_quantize(cls, properties):
        kwargs = properties['kwargs']

        if 'quantize' in properties and not properties['quantize']:
            return properties

        # device map is not required for lmi dist and
        if properties['rolling_batch'] == RollingBatchEnum.lmidist or \
                properties['rolling_batch'] == RollingBatchEnum.vllm:
            return properties

        if properties[
                'quantize'].value == HFQuantizeMethods.bitsandbytes8.value:
            if "device_map" not in kwargs:
                raise ValueError(
                    "device_map should be set when load_in_8bit is set")
            kwargs["load_in_8bit"] = properties['load_in_8bit']
            properties['quantize'] = HFQuantizeMethods.bitsandbytes8
        if properties[
                'quantize'].value == HFQuantizeMethods.bitsandbytes4.value:
            if "device_map" not in kwargs:
                raise ValueError(
                    "device_map should set when load_in_4bit is set")
            kwargs["load_in_4bit"] = properties['load_in_4bit']
            properties['quantize'] = HFQuantizeMethods.bitsandbytes4

        properties['kwargs'] = kwargs
        return properties

    @root_validator()
    def construct_kwargs_dtype(cls, properties):

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
                return dtype
            raise ValueError(f"Invalid data type: {dtype}")

        kwargs = properties['kwargs']

        if properties.get('data_type'):
            logging.warning('option.data_type is deprecated.'
                            'Please use option.dtype')
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties['data_type'].lower())
        if properties.get('dtype'):
            kwargs["torch_dtype"] = get_torch_dtype_from_str(
                properties['dtype'].lower())

        properties['kwargs'] = kwargs
        return properties

    @root_validator()
    def set_device_mpi(cls, properties):
        if properties['rolling_batch'].value != RollingBatchEnum.disable.value:
            if properties['is_mpi']:
                properties['device'] = int(os.getenv("LOCAL_RANK", 0))
        return properties
