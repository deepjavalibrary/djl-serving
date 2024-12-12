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
import dataclasses
from typing import Optional, Any, Mapping, Tuple, Dict

from pydantic import field_validator, model_validator, ConfigDict
from vllm import EngineArgs

from djl_python.properties_manager.properties import Properties

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}


class VllmRbProperties(Properties):
    engine: Optional[str] = None
    # The following configs have different names in DJL compared to vLLM
    quantize: Optional[str] = None
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    max_rolling_batch_prefill_tokens: Optional[int] = None
    cpu_offload_gb_per_gpu: Optional[int] = 0
    # The following configs have different defaults, or additional processing in DJL compared to vLLM
    dtype: str = "auto"
    max_loras: Optional[int] = 4
    long_lora_scaling_factors: Optional[Tuple[float, ...]] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None

    # Neuron vLLM properties
    device: Optional[str] = None
    preloaded_model: Optional[Any] = None
    generation_config: Optional[Any] = None

    # This allows generic vllm engine args to be passed in and set with vllm
    model_config = ConfigDict(extra='allow')

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

    @model_validator(mode='after')
    def validate_pipeline_parallel(self):
        if self.pipeline_parallel_degree != 1:
            raise ValueError(
                "Pipeline parallelism is not supported in vLLM's LLMEngine used in rolling_batch implementation"
            )
        return self

    def handle_lmi_vllm_config_conflicts(self, additional_vllm_engine_args):

        def djl_config_conflicts_with_vllm_config(lmi_config_name,
                                                  vllm_config_name) -> bool:
            # TODO: We may be able to refactor this to throw the ValueError directly from this method.
            # The errors are slightly different depending on the specific configs, so for now we keep
            # the exception separate in favor of better, more specific client errors
            lmi_config_val = self.__getattribute__(lmi_config_name)
            vllm_config_val = additional_vllm_engine_args.get(vllm_config_name)
            if vllm_config_val is not None and lmi_config_val is not None:
                return lmi_config_val != vllm_config_val
            return False

        if djl_config_conflicts_with_vllm_config("quantize", "quantization"):
            raise ValueError(
                "Both the DJL quantize config, and vllm quantization configs have been set with conflicting values."
                "Only set the DJL quantize config")
        if djl_config_conflicts_with_vllm_config("tensor_parallel_degree",
                                                 "tensor_parallel_size"):
            raise ValueError(
                "Both the DJL tensor_parallel_degree and vllm tensor_parallel_size configs have been set with conflicting values."
                "Only set the DJL tensor_parallel_degree config")
        if djl_config_conflicts_with_vllm_config("pipeline_parallel_degree",
                                                 "pipeline_parallel_size"):
            raise ValueError(
                "Both the DJL pipeline_parallel_degree and vllm pipeline_parallel_size configs have been set with conflicting values."
                "Only set the DJL pipeline_parallel_degree config")
        if djl_config_conflicts_with_vllm_config(
                "max_rolling_batch_prefill_tokens", "max_num_batched_tokens"):
            raise ValueError(
                "Both the DJL max_rolling_batch_prefill_tokens and vllm max_num_batched_tokens configs have been set with conflicting values."
                "Only set one of these configurations")
        if djl_config_conflicts_with_vllm_config("cpu_offload_gb_per_gpu",
                                                 "cpu_offload_gb"):
            raise ValueError(
                "Both the DJL cpu_offload_gb_per_gpu and vllm cpu_offload_gb configs have been set with conflicting values."
                "Only set one of these configurations")

    def get_engine_args(self) -> EngineArgs:
        additional_vllm_engine_args = self.get_additional_vllm_engine_args()
        self.handle_lmi_vllm_config_conflicts(additional_vllm_engine_args)
        max_model_len = additional_vllm_engine_args.pop("max_model_len", None)
        if self.device == 'neuron':
            return EngineArgs(
                model=self.model_id_or_path,
                preloaded_model=self.preloaded_model,
                tensor_parallel_size=self.tensor_parallel_degree,
                pipeline_parallel_size=self.pipeline_parallel_degree,
                dtype=DTYPE_MAPPER[self.dtype],
                max_num_seqs=self.max_rolling_batch_size,
                block_size=max_model_len,
                max_model_len=max_model_len,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                device=self.device,
                generation_config=self.generation_config,
                **additional_vllm_engine_args,
            )
        return EngineArgs(
            model=self.model_id_or_path,
            tensor_parallel_size=self.tensor_parallel_degree,
            pipeline_parallel_size=self.pipeline_parallel_degree,
            dtype=DTYPE_MAPPER[self.dtype],
            max_model_len=max_model_len,
            quantization=self.quantize,
            max_num_batched_tokens=self.max_rolling_batch_prefill_tokens,
            max_loras=self.max_loras,
            long_lora_scaling_factors=self.long_lora_scaling_factors,
            cpu_offload_gb=self.cpu_offload_gb_per_gpu,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            **additional_vllm_engine_args,
        )

    def get_additional_vllm_engine_args(self) -> Dict[str, Any]:
        all_engine_args = EngineArgs.__annotations__
        return {
            arg: val
            for arg, val in self.__pydantic_extra__.items()
            if arg in all_engine_args
        }
