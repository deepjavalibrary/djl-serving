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
from typing import Optional

from pydantic.v1.class_validators import validator, root_validator

from djl_python.properties_manager.properties import Properties


class LmiDistQuantizeMethods(str, Enum):
    awq = 'awq'
    gptq = 'gptq'
    squeezellm = 'squeezellm'


class LmiDistRbProperties(Properties):
    engine: Optional[str] = None
    dtype: Optional[str] = "auto"
    load_format: Optional[str] = "auto"
    quantize: Optional[LmiDistQuantizeMethods] = None
    tensor_parallel_degree: Optional[int] = None
    max_rolling_batch_prefill_tokens: Optional[int] = None
    # Adjustable prefix model length for certain 32k or longer model
    max_model_len: Optional[int] = None
    # TODO: change Enforce eager to False once SageMaker driver issue resolved
    enforce_eager: Optional[bool] = False
    # TODO: this default may change with different vLLM versions
    # TODO: try to get good default from vLLM to prevent revisiting
    # TODO: last time check: vllm 0.3.1
    gpu_memory_utilization: Optional[float] = 0.9
    # TODO: speculative decoding changes
    speculative_draft_model: Optional[str] = None
    speculative_length: int = 5
    draft_model_tp_size: int = 1
    record_acceptance_rate: Optional[bool] = False
    enable_lora: Optional[bool] = False
    max_loras: Optional[int] = 4
    max_lora_rank: Optional[int] = 16
    lora_extra_vocab_size: Optional[int] = 256
    max_cpu_loras: Optional[int] = None

    @root_validator(skip_on_failure=True)
    def validate_mpi(cls, properties):
        if not properties["is_mpi"]:
            raise AssertionError(
                f"Need MPI engine to start lmi-dist RollingBatcher")
        return properties

    @root_validator(skip_on_failure=True)
    def validate_speculative_and_lora(cls, properties):
        if properties["enable_lora"] and properties["speculative_draft_model"]:
            raise AssertionError(
                f"Cannot enable lora and speculative decoding at the same time"
            )
        return properties
