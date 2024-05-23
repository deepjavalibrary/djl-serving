#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
from enum import Enum
from typing import Optional, Any

from pydantic import field_validator

from djl_python.properties_manager.properties import Properties


class VllmQuantizeMethods(str, Enum):
    awq = 'awq'
    gptq = 'gptq'
    squeezellm = 'squeezellm'


class VllmRbProperties(Properties):
    engine: Optional[str] = None
    dtype: Optional[str] = "auto"
    load_format: Optional[str] = "auto"
    quantize: Optional[VllmQuantizeMethods] = None
    tensor_parallel_degree: Optional[int] = None
    max_rolling_batch_prefill_tokens: Optional[int] = None
    # Adjustable prefix model length for certain 32k or longer model
    max_model_len: Optional[int] = None
    enforce_eager: Optional[bool] = False
    # TODO: this default may change with different vLLM versions
    # TODO: try to get good default from vLLM to prevent revisiting
    # TODO: last time check: vllm 0.3.1
    gpu_memory_utilization: Optional[float] = 0.9
    enable_lora: Optional[bool] = False
    max_loras: Optional[int] = 4
    max_lora_rank: Optional[int] = 16
    lora_extra_vocab_size: Optional[int] = 256
    max_cpu_loras: Optional[int] = None
    # Neuron vLLM properties
    device: Optional[str] = None
    preloaded_model: Optional[Any] = None

    @field_validator('engine')
    def validate_engine(cls, engine):
        if engine != "Python":
            raise AssertionError(
                f"Need python engine to start vLLM RollingBatcher")
        return engine