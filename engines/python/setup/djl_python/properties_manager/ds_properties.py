import json
import logging
import os
from enum import Enum

import torch
from typing import Optional, Any

from pydantic import root_validator, validator, Field

from djl_python.properties_manager.properties import Properties, RollingBatchEnum


class DsQuantizeMethods(str, Enum):
    smoothquant = 'smoothquant'
    dynamicint8 = 'dynamic_int8'


SUPPORTED_QUANTIZATION_MODE = [
    DsQuantizeMethods.smoothquant.value, DsQuantizeMethods.dynamicint8.value
]
DS_SUPPORTED_ROLLING_BATCH_TYPES = [
    RollingBatchEnum.auto.value, RollingBatchEnum.deepspeed.value
]


class DeepSpeedProperties(Properties):
    """ Configures DeepSpeed related properties """

    quantize: Optional[DsQuantizeMethods] = None
    dtype: Optional[Any] = None
    max_tokens: int = 1024
    low_cpu_mem_usage: bool = True
    task: Optional[str] = None
    enable_cuda_graph: bool = False
    triangular_masking: bool = True
    device: int = 0
    smoothquant_alpha: Optional[float]
    return_tuple: bool = True
    training_mp_size: Optional[int] = 1
    checkpoint: Optional[str] = None
    save_mp_checkpoint_path: Optional[str] = None
    ds_config: Optional[Any] = Field(default={}, alias='deepspeed_config_path')

    @validator('device', always=True)
    def set_device(cls, device):
        device = int(os.getenv("LOCAL_RANK", 0))
        return device

    @validator('quantize', pre=True)
    def validate_quantize(cls, quantize, values):
        if quantize not in SUPPORTED_QUANTIZATION_MODE:
            logging.warning(
                f"DeepSpeed does not currently support quantization mode: ${quantize}, "
                f"this setting will be ignored.")
            return None
        elif quantize == DsQuantizeMethods.smoothquant.value and values.get(
                "dtype") == 'bf16':
            raise ValueError(
                "dtype should not be bf16 while using smoothquant")

        return quantize

    @validator('smoothquant_alpha')
    def validate_smoothquant_alpha(cls, smoothquant_alpha) -> dict:
        try:
            if 0 <= float(smoothquant_alpha) <= 1:
                return smoothquant_alpha
            else:
                raise ValueError(
                    f"${smoothquant_alpha} is not between 0 and 1.")
        except ValueError:
            raise ValueError(
                f"${smoothquant_alpha} cannot convert to float number.")

    @validator('ds_config', pre=True)
    def set_ds_config(cls, deepspeed_config_path):
        with open(deepspeed_config_path, "r") as f:
            return json.load(f)

    @validator('rolling_batch', pre=True)
    def validate_rolling_batch(cls, rolling_batch) -> bool:
        if rolling_batch == RollingBatchEnum.disable.value:
            return rolling_batch
        if rolling_batch not in DS_SUPPORTED_ROLLING_BATCH_TYPES:
            raise ValueError(
                f"deepspeed engine only supports "
                f"rolling batch type {DS_SUPPORTED_ROLLING_BATCH_TYPES}.")
        return rolling_batch

    @root_validator()
    def set_dtype(cls, properties):

        def get_default_dtype(quantize_mode: str):
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported() \
                        and quantize_mode != DsQuantizeMethods.smoothquant.value:
                    return "bf16"
                return "fp16"
            return "fp32"

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

        quantize_str = None
        if properties.get('quantize'):
            quantize_str = properties['quantize'].value

        default_dtype = get_default_dtype(quantize_str)

        properties['dtype'] = get_torch_dtype_from_str(
            properties.get('dtype', default_dtype))
        return properties

    @root_validator()
    def construct_ds_config(cls, properties):
        if not properties.get("ds_config"):
            ds_config = {
                "tensor_parallel": {
                    "tp_size": properties['tensor_parallel_degree']
                },
                "enable_cuda_graph":
                properties['enable_cuda_graph'],
                "triangular_masking":
                properties['triangular_masking'],
                "return_tuple":
                properties["return_tuple"],
                "training_mp_size":
                int(properties["training_mp_size"]),
                "max_tokens":
                properties['max_tokens'],
                "save_mp_checkpoint_path":
                properties.get("save_mp_checkpoint_path")
            }

            if "checkpoint" in properties and properties['checkpoint']:
                ds_config["checkpoint"] = os.path.join(
                    properties['model_id_or_path'],
                    properties.get("checkpoint"))
                ds_config["base_dir"] = properties['model_id_or_path']
                if properties['dtype'] is None:
                    raise ValueError(
                        "dtype should also be provided for checkpoint loading")

            if 'quantize' in properties and properties['quantize']:
                ds_config['dynamic_quant'] = {
                    'enabled': True,
                    'use_cutlass': False
                }
                if properties[
                        'quantize'] == DsQuantizeMethods.smoothquant.value:
                    smoothing_value = {'smooth': True, 'calibrate': True}
                    if 'smoothquant_alpha' in properties and properties[
                            'smoothquant_alpha']:
                        smoothing_value['alpha'] = properties[
                            'smoothquant_alpha']
                    ds_config['smoothing'] = smoothing_value

            properties['ds_config'] = ds_config
        return properties
