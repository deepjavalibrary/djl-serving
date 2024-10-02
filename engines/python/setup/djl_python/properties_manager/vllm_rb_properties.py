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
from typing import Optional, Any, Mapping

from pydantic import field_validator

from djl_python.properties_manager.properties import Properties


class VllmQuantizeMethods(str, Enum):
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
    max_logprobs: Optional[int] = 20
    enable_chunked_prefill: Optional[bool] = False
    cpu_offload_gb_per_gpu: Optional[int] = 0
    enable_prefix_caching: Optional[bool] = False
    disable_sliding_window: Optional[bool] = False
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    use_v2_block_manager: bool = False

    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    @field_validator('engine')
    def validate_engine(cls, engine):
        if engine != "Python":
            raise AssertionError(
                f"Need python engine to start vLLM RollingBatcher")
        return engine

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
