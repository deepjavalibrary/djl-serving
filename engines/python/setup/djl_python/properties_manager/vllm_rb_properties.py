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
import logging
from typing import Optional, Any, Dict, Tuple
from pydantic import field_validator, model_validator, ConfigDict, Field
from vllm import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import StoreBoolean

from djl_python.properties_manager.properties import Properties

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}


class VllmRbProperties(Properties):
    engine: Optional[str] = None
    # The following configs have different names in DJL compared to vLLM, we only accept DJL name currently
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    # The following configs have different names in DJL compared to vLLM, either is accepted
    quantize: Optional[str] = Field(alias="quantization", default=None)
    max_rolling_batch_prefill_tokens: Optional[int] = Field(
        alias="max_num_batched_tokens", default=None)
    cpu_offload_gb_per_gpu: Optional[float] = Field(alias="cpu_offload_gb",
                                                    default=None)
    # The following configs have different defaults, or additional processing in DJL compared to vLLM
    dtype: str = "auto"
    max_loras: int = 4
    long_lora_scaling_factors: Optional[Tuple[float, ...]] = None

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

    @model_validator(mode='after')
    def validate_pipeline_parallel(self):
        if self.pipeline_parallel_degree != 1:
            raise ValueError(
                "Pipeline parallelism is not supported in vLLM's LLMEngine used in rolling_batch implementation"
            )
        return self

    @field_validator('long_lora_scaling_factors', mode='before')
    # TODO: processing of this field is broken in vllm via from_cli_args
    # we should upstream a fix for this to vllm
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

    def handle_lmi_vllm_config_conflicts(self, additional_vllm_engine_args):

        def validate_potential_lmi_vllm_config_conflict(
                lmi_config_name, vllm_config_name):
            lmi_config_val = self.__getattribute__(lmi_config_name)
            vllm_config_val = additional_vllm_engine_args.get(vllm_config_name)
            if vllm_config_val is not None and lmi_config_val is not None:
                if vllm_config_val != lmi_config_val:
                    raise ValueError(
                        f"Both the DJL {lmi_config_val}={lmi_config_val} and vLLM {vllm_config_name}={vllm_config_val} configs have been set with conflicting values."
                        f"We currently only accept the DJL config {lmi_config_val}, please remove the vllm {vllm_config_name} configuration."
                    )

        validate_potential_lmi_vllm_config_conflict("tensor_parallel_degree",
                                                    "tensor_parallel_size")
        validate_potential_lmi_vllm_config_conflict("pipeline_parallel_degree",
                                                    "pipeline_parallel_size")
        validate_potential_lmi_vllm_config_conflict("max_rolling_batch_size",
                                                    "max_num_seqs")

    def generate_vllm_engine_arg_dict(self,
                                      passthrough_vllm_engine_args) -> dict:
        vllm_engine_args = {
            'model': self.model_id_or_path,
            'tensor_parallel_size': self.tensor_parallel_degree,
            'pipeline_parallel_size': self.pipeline_parallel_degree,
            'max_num_seqs': self.max_rolling_batch_size,
            'dtype': DTYPE_MAPPER[self.dtype],
            'revision': self.revision,
            'max_loras': self.max_loras,
            'enable_lora': self.enable_lora,
        }
        if self.quantize is not None:
            vllm_engine_args['quantization'] = self.quantize
        if self.max_rolling_batch_prefill_tokens is not None:
            vllm_engine_args[
                'max_num_batched_tokens'] = self.max_rolling_batch_prefill_tokens
        if self.cpu_offload_gb_per_gpu is not None:
            vllm_engine_args['cpu_offload_gb'] = self.cpu_offload_gb_per_gpu
        if self.device is not None:
            vllm_engine_args['device'] = self.device
        if self.preloaded_model is not None:
            vllm_engine_args['preloaded_model'] = self.preloaded_model
        if self.generation_config is not None:
            vllm_engine_args['generation_config'] = self.generation_config
        vllm_engine_args.update(passthrough_vllm_engine_args)
        return vllm_engine_args

    def get_engine_args(self) -> EngineArgs:
        additional_vllm_engine_args = self.get_additional_vllm_engine_args()
        self.handle_lmi_vllm_config_conflicts(additional_vllm_engine_args)
        vllm_engine_arg_dict = self.generate_vllm_engine_arg_dict(
            additional_vllm_engine_args)
        logging.debug(
            f"Construction vLLM engine args from the following DJL configs: {vllm_engine_arg_dict}"
        )
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        args_list = self.construct_vllm_args_list(vllm_engine_arg_dict, parser)
        args = parser.parse_args(args=args_list)
        engine_args = EngineArgs.from_cli_args(args)
        # we have to do this separately because vllm converts it into a string
        engine_args.long_lora_scaling_factors = self.long_lora_scaling_factors
        return engine_args

    def get_additional_vllm_engine_args(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__pydantic_extra__.items()
            if k in EngineArgs.__annotations__
        }

    def construct_vllm_args_list(self, vllm_engine_args: dict,
                                 parser: FlexibleArgumentParser):
        # Modified from https://github.com/vllm-project/vllm/blob/v0.6.4/vllm/utils.py#L1258
        args_list = []
        store_boolean_arguments = {
            action.dest
            for action in parser._actions if isinstance(action, StoreBoolean)
        }
        for engine_arg, engine_arg_value in vllm_engine_args.items():
            if str(engine_arg_value).lower() in {
                    'true', 'false'
            } and engine_arg not in store_boolean_arguments:
                if str(engine_arg_value).lower() == 'true':
                    args_list.append(f"--{engine_arg}")
            else:
                args_list.append(f"--{engine_arg}={engine_arg_value}")
        return args_list
