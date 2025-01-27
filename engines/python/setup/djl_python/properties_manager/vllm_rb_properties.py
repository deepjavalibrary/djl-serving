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
import ast
import json
from enum import Enum
from typing import Optional, Any, Mapping, Tuple, Dict

from pydantic import field_validator, model_validator

from djl_python.properties_manager.properties import Properties


class VllmRbProperties(Properties):
    engine: Optional[str] = None
    dtype: Optional[str] = "auto"
    load_format: Optional[str] = "auto"
    quantize: Optional[str] = None
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
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
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float, ...]] = None
    lora_dtype: Optional[str] = 'auto'
    max_cpu_loras: Optional[int] = None

    # Neuron vLLM properties
    device: Optional[str] = None
    preloaded_model: Optional[Any] = None
    generation_config: Optional[Any] = None
    override_neuron_config: Optional[Dict] = None

    max_logprobs: Optional[int] = 20
    enable_chunked_prefill: Optional[bool] = None
    cpu_offload_gb_per_gpu: Optional[int] = 0
    enable_prefix_caching: Optional[bool] = False
    disable_sliding_window: Optional[bool] = False
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    use_v2_block_manager: bool = False
    tokenizer_mode: str = 'auto'

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

    @field_validator('long_lora_scaling_factors', mode='before')
    def validate_long_lora_scaling_factors(cls, val):
        if isinstance(val, str):
            val = ast.literal_eval(val)
        if not isinstance(val, tuple):
            if isinstance(val, list):
                val = tuple(float(v) for v in val)
            elif isinstance(val, float):
                val = (val, )
            elif isinstance(val, int):
                val = (float(val), )
            else:
                raise ValueError(
                    "long_lora_scaling_factors must be convertible to a tuple of floats."
                )
        return val

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

    @field_validator('override_neuron_config', mode="before")
    def validate_override_neuron_config(cls, val):
        if isinstance(val, str):
            neuron_config = ast.literal_eval(val)
            if not isinstance(neuron_config, dict):
                raise ValueError(
                    f"Invalid json format for override_neuron_config")
            return neuron_config
        elif isinstance(val, Dict):
            return val
        else:
            raise ValueError("Invalid json format for override_neuron_config")

    @model_validator(mode='after')
    def validate_speculative_model(self):
        if self.speculative_model is not None and not self.use_v2_block_manager:
            raise ValueError(
                "Speculative decoding requires usage of the V2 block manager. Enable it with option.use_v2_block_manager=true."
            )
        return self

    @model_validator(mode='after')
    def validate_pipeline_parallel(self):
        if self.pipeline_parallel_degree != 1:
            raise ValueError(
                "Pipeline parallelism is not supported in vLLM's LLMEngine used in rolling_batch implementation"
            )
        return self
