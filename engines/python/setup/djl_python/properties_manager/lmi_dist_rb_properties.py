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
from typing import Optional, Mapping

from pydantic import model_validator, field_validator

from djl_python.properties_manager.properties import Properties


class LmiDistQuantizeMethods(str, Enum):
    awq = 'awq'
    deepspeedfp = 'deepspeedfp'
    fp8 = 'fp8'
    fbgemm_fp8 = 'fbgemm_fp8'
    gptq = 'gptq'
    gptq_marlin = 'gptq_marlin'
    gptq_marlin_24 = 'gptq_marlin_24'
    awq_marlin = 'awq_marlin'
    marlin = 'marlin'
    squeezellm = 'squeezellm'

class LmiDistLoadFormats(str, Enum):
    sagemaker_fast_model_loader = 'sagemaker_fast_model_loader'

class LmiDistRbProperties(Properties):
    engine: Optional[str] = None
    dtype: Optional[str] = "auto"
    load_format: Optional[str] = "auto"
    quantize: Optional[LmiDistQuantizeMethods] = None
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
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
    speculative_length: int = 4
    draft_model_tp_size: int = 1
    record_acceptance_rate: Optional[bool] = False
    speculative_telemetry: Optional[bool] = True
    enable_lora: Optional[bool] = False
    max_loras: Optional[int] = 4
    max_lora_rank: Optional[int] = 16
    lora_extra_vocab_size: Optional[int] = 256
    max_cpu_loras: Optional[int] = None
    max_logprobs: Optional[int] = 20
    enable_chunked_prefill: Optional[bool] = None
    cpu_offload_gb_per_gpu: Optional[int] = 0
    enable_prefix_caching: Optional[bool] = False
    disable_sliding_window: Optional[bool] = False
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    use_passive_workers: Optional[bool] = True
    tokenizer_mode: str = 'auto'

    @model_validator(mode='after')
    def validate_mpi(self):
        if not self.mpi_mode:
            raise AssertionError(
                f"Need MPI engine to start lmi-dist RollingBatcher")
        return self

    @model_validator(mode='after')
    def validate_speculative_and_lora(self):
        if self.enable_lora and self.speculative_draft_model:
            raise AssertionError(
                f"Cannot enable lora and speculative decoding at the same time"
            )
        return self

    @model_validator(mode='after')
    def validate_speculative_and_fml(self):
        if self.load_format == LmiDistLoadFormats.sagemaker_fast_model_loader and self.speculative_draft_model:
            raise AssertionError(
                f"Cannot enable sagemaker_fast_model_loader and speculative decoding at the same time"
            )
        return self

    @field_validator('limit_mm_per_prompt', mode="before")
    def validate_limit_mm_per_prompt(cls, val) -> Mapping[str, int]:
        out_dict: Dict[str, int] = {}
        for item in val.split(","):
            kv_parts = [part.lower().strip() for part in item.split("=")]
            if len(kv_parts) != 2:
                raise ValueError("Each item should be in the form key=value")
            key, value = kv_parts

            try:
                parsed_value = int(value)
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse value of item {key}={value}") from e

            if key in out_dict and out_dict[key] != parsed_value:
                raise ValueError(
                    f"Conflicting values specified for key: {key}")
            out_dict[key] = parsed_value
        return out_dict
