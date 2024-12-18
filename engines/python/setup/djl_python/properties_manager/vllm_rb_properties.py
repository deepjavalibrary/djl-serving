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
import argparse
from dataclasses import asdict
from typing import Optional, Any, Mapping, Tuple, Dict
from pydantic import field_validator, model_validator, ConfigDict
from vllm import EngineArgs
from vllm.utils import FlexibleArgumentParser

from djl_python.properties_manager.properties import Properties

DEFAULT_ENGINE_ARGS = asdict(EngineArgs())

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
        if djl_config_conflicts_with_vllm_config("max_rolling_batch_size",
                                                 "max_num_seqs"):
            raise ValueError(
                "Both the DJL max_rolling_batch_size and vllm max_num_seqs configs have been set with conflicting values."
                "Only set the DJL max_rolling_batch_size config")

    def generate_vllm_engine_arg_dict(self,
                                      passthrough_vllm_engine_args) -> dict:
        # We use the full set of engine args here in order for the EngineArgs.from_cli_args call to work since
        # it requires all engine args at least be present.
        # TODO: We may want to upstream a change to vllm here to make this a bit nicer for us
        vllm_engine_args = DEFAULT_ENGINE_ARGS.copy()
        # For the following configs, we only accept the LMI name currently (may be same as vllm config name)
        vllm_engine_args['model'] = self.model_id_or_path
        vllm_engine_args['tensor_parallel_size'] = self.tensor_parallel_degree
        vllm_engine_args[
            'pipeline_parallel_size'] = self.pipeline_parallel_degree
        vllm_engine_args['max_num_seqs'] = self.max_rolling_batch_size
        vllm_engine_args['dtype'] = DTYPE_MAPPER[self.dtype]
        vllm_engine_args['trust_remote_code'] = self.trust_remote_code
        vllm_engine_args['revision'] = self.revision
        vllm_engine_args['max_loras'] = self.max_loras
        # For these configs, either the LMI or vllm name is ok
        vllm_engine_args['quantization'] = passthrough_vllm_engine_args.pop(
            'quantization', self.quantize)
        vllm_engine_args[
            'max_num_batched_tokens'] = passthrough_vllm_engine_args.pop(
                'max_num_batched_tokens',
                self.max_rolling_batch_prefill_tokens)
        vllm_engine_args['cpu_offload_gb'] = passthrough_vllm_engine_args.pop(
            'cpu_offload_gb', self.cpu_offload_gb_per_gpu)
        # Neuron specific configs
        if self.device == 'neuron':
            vllm_engine_args['device'] = self.device
            vllm_engine_args['preloaded_model'] = self.preloaded_model
            vllm_engine_args['generation_config'] = self.generation_config
        vllm_engine_args.update(passthrough_vllm_engine_args)
        return vllm_engine_args

    def get_engine_args(self) -> EngineArgs:
        additional_vllm_engine_args = self.get_additional_vllm_engine_args()
        self.handle_lmi_vllm_config_conflicts(additional_vllm_engine_args)
        vllm_engine_arg_dict = self.generate_vllm_engine_arg_dict(
            additional_vllm_engine_args)
        args_list = [f"--{k}={v}" for k, v in vllm_engine_arg_dict.items()]
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        args = parser.parse_args(args=args_list)
        return EngineArgs.from_cli_args(args)

    def get_additional_vllm_engine_args(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__pydantic_extra__.items()
            if k in DEFAULT_ENGINE_ARGS
        }
