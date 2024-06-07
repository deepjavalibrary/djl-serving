import logging
import os
from enum import Enum
from typing import Optional

import torch

from pydantic import field_validator, model_validator

from djl_python.properties_manager.properties import Properties, RollingBatchEnum


class HFQuantizeMethods(str, Enum):
    # added for backward compatibility lmi-dist
    bitsandbytes = 'bitsandbytes'
    gptq = 'gptq'

    # huggingface
    bitsandbytes4 = 'bitsandbytes4'
    bitsandbytes8 = 'bitsandbytes8'

    # TODO remove this after refactor of all handlers
    # supported by vllm
    awq = 'awq'
    deepspeedfp = 'deepspeedfp'


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
    save_mp_checkpoint_path: Optional[str] = None

    device: Optional[str] = None
    kwargs: Optional[dict] = {}
    data_type: Optional[str] = None

    @field_validator('load_in_4bit')
    def validate_load_in_4bit(cls, load_in_4bit):
        logging.warning('option.load_in_4bit is deprecated. '
                        'Kindly use option.quantize=bitsandbytes4 instead')
        return load_in_4bit

    @field_validator('load_in_8bit')
    def validate_load_in_8bit(cls, load_in_8bit):
        logging.warning('option.load_in_8bit is deprecated. '
                        'Kindly use option.quantize=bitsandbytes8 instead')
        return load_in_8bit

    @model_validator(mode='after')
    def set_quantize_for_backward_compatibility(self):
        if self.load_in_4bit:
            self.quantize = HFQuantizeMethods.bitsandbytes4
        elif self.load_in_8bit:
            self.quantize = HFQuantizeMethods.bitsandbytes8

        # TODO remove this after refactor of all handlers
        # parsing bitsandbytes8, so it can be directly passed to lmi dist model loader.
        if self.quantize == HFQuantizeMethods.bitsandbytes8 \
                and self.rolling_batch == RollingBatchEnum.lmidist:
            self.quantize = HFQuantizeMethods.bitsandbytes
        return self

    @model_validator(mode='after')
    def set_device(self):
        if self.device_id >= 0:
            self.device = f"cuda:{self.device_id}"
        else:
            self.device = None
        return self

    @model_validator(mode='after')
    def construct_kwargs(self):
        self.kwargs['trust_remote_code'] = self.trust_remote_code
        if self.low_cpu_mem_usage:
            self.kwargs["low_cpu_mem_usage"] = self.low_cpu_mem_usage
        if self.revision:
            self.kwargs["revision"] = self.revision

        # TODO remove this after refactor of all handlers
        if self.rolling_batch.value != RollingBatchEnum.disable.value:
            if self.waiting_steps:
                self.kwargs["waiting_steps"] = self.waiting_steps

        return self

    @model_validator(mode='after')
    def construct_kwargs_device_map(self):
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map
        if self.device_map:
            self.kwargs["device_map"] = self.device_map
            self.device = None
            logging.info(f"Using device map {self.device_map}")
        elif self.tensor_parallel_degree > 0 and torch.cuda.device_count() > 0:
            self.kwargs["device_map"] = "auto"
            self.device = None
            world_size = torch.cuda.device_count()
            assert world_size == self.tensor_parallel_degree, \
                f"TP degree ({self.tensor_parallel_degree}) doesn't match available GPUs ({world_size})"
            logging.info(f"Using {world_size} gpus")
        return self

    @model_validator(mode='after')
    def construct_kwargs_quantize(self):
        if not self.quantize:
            return self

        # TODO remove this after refactor of all handlers
        # device map is not required for lmi dist and vllm
        if self.rolling_batch in {
                RollingBatchEnum.lmidist,
                RollingBatchEnum.vllm,
        }:
            return self

        if self.quantize.value == HFQuantizeMethods.bitsandbytes8.value:
            if "device_map" not in self.kwargs:
                raise ValueError(
                    "device_map should be set when load_in_8bit is set")
            self.kwargs["load_in_8bit"] = True
        if self.quantize.value == HFQuantizeMethods.bitsandbytes4.value:
            if "device_map" not in self.kwargs:
                raise ValueError(
                    "device_map should set when load_in_4bit is set")
            self.kwargs["load_in_4bit"] = True

        return self

    @model_validator(mode='after')
    def construct_kwargs_dtype(self):
        if self.dtype:
            self.kwargs["torch_dtype"] = get_torch_dtype_from_str(
                self.dtype.lower())
        return self

    @model_validator(mode='after')
    def set_device_mpi(self):
        if self.rolling_batch.value != RollingBatchEnum.disable.value:
            if self.mpi_mode:
                self.device = str(os.getenv("LOCAL_RANK", 0))
        return self
