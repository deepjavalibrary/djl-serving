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
import json
import logging
from pydantic import ConfigDict
from typing import Optional

from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import QuantConfig, CalibConfig, QuantAlgo

from djl_python.properties_manager.properties import Properties

logger = logging.getLogger(__name__)


class TensorRtLlmProperties(Properties):

    # in our implementation, we handle compilation/quantization ahead of time.
    # we do not expose build_config, quant_config, calib_config here as those get handled by
    # the compilation in trt_llm_partition.py. We do it this way so that users can completely build the complete
    # trt engine ahead of time via that script. If provided just a HF model id, then that script gets invoked,
    # does compilation/quantization and generates engines that will get loaded here. We are only exposing
    # runtime knobs here.

    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    skip_tokenizer_init: bool = False
    dtype: str = 'auto'
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    moe_tensor_parallel_size: Optional[int] = None
    moe_expert_parallel_size: Optional[int] = None
    enable_attention_dp: bool = False
    auto_parallel: bool = False
    auto_parallel_world_size: Optional[int] = None
    load_format: str = 'auto'
    enable_chunked_prefill: bool = False
    guided_decoding_backend: Optional[str] = None
    iter_stats_max_iterations: Optional[int] = None
    request_stats_max_iterations: Optional[int] = None
    embedding_parallel_mode: str = 'SHARDING_ALONG_VOCAB'
    fast_build: bool = False
    # different default! allows for faster loading on worker restart
    enable_build_cache: bool = True
    batching_type: Optional[None] = None
    normalize_log_probs: bool = False
    gather_generation_logits: bool = False
    extended_runtime_perf_knob_config: Optional[None] = None
    max_batch_size: Optional[int] = None
    max_input_len: int = 1024
    max_seq_len: Optional[int] = None
    max_beam_width: int = 1
    max_num_tokens: Optional[int] = None
    backend: Optional[str] = None

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    def get_kv_cache_config(self) -> Optional[KvCacheConfig]:
        kv_cache_config = {}
        if "enable_block_reuse" in self.__pydantic_extra__:
            kv_cache_config["enable_block_reuse"] = self.__pydantic_extra__[
                "enable_block_reuse"].lower() == "true"
        if "max_tokens" in self.__pydantic_extra__:
            kv_cache_config["max_tokens"] = int(
                self.__pydantic_extra__["max_tokens"])
        if "max_attention_window" in self.__pydantic_extra__:
            kv_cache_config["max_attention_window"] = json.loads(
                self.__pydantic_extra__["max_attention_window"])
        if "sink_token_length" in self.__pydantic_extra__:
            kv_cache_config["sink_token_length"] = int(
                self.__pydantic_extra__["sink_token_length"])
        if "free_gpu_memory_fraction" in self.__pydantic_extra__:
            kv_cache_config["free_gpu_memory_fraction"] = float(
                self.__pydantic_extra__["free_gpu_memory_fraction"])
        if "host_cache_size" in self.__pydantic_extra__:
            kv_cache_config["host_cache_size"] = int(
                self.__pydantic_extra__["host_cache_size"])
        if "onboard_blocks" in self.__pydantic_extra__:
            kv_cache_config["onboard_blocks"] = self.__pydantic_extra__[
                "onboard_blocks"].lower() == "true"
        if "cross_kv_cache_fraction" in self.__pydantic_extra__:
            kv_cache_config["cross_kv_cache_fraction"] = float(
                self.__pydantic_extra__["cross_kv_cache_fraction"])
        if "secondary_offload_min_priority" in self.__pydantic_extra__:
            kv_cache_config["secondary_offload_min_priority"] = int(
                self.__pydantic_extra__["secondary_offload_min_priority"])
        if "event_buffer_max_size" in self.__pydantic_extra__:
            kv_cache_config["event_buffer_max_size"] = int(
                self.__pydantic_extra__["event_buffer_max_size"])

        return KvCacheConfig(**kv_cache_config)

    def get_pytorch_config(self) -> Optional[PyTorchConfig]:
        if self.backend != 'pytorch':
            return None
        # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/pytorch/quickstart_advanced.py#L107
        pytorch_config = {
            "enable_overlap_scheduler":
            self.__pydantic_extra__.get('enable_overlap_scheduler',
                                        'false').lower() == 'true',
            "kv_cache_dtype":
            self.__pydantic_extra__.get('kv_cache_dtype', 'auto'),
            "attn_backend":
            self.__pydantic_extra__.get('attn_backend', 'TRTLLM'),
            'use_cuda_graph':
            self.__pydantic_extra__.get('use_cuda_graph',
                                        'false').lower() == 'true',
            'load_format':
            self.__pydantic_extra__.get('load_format', 'auto'),
            'moe_backend':
            self.__pydantic_extra__.get('moe_backend', 'CUTLASS')
        }
        return PyTorchConfig(**pytorch_config)

    def get_llm_kwargs(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "tokenizer_mode": self.tokenizer_mode,
            "skip_tokenizer_init": self.skip_tokenizer_init,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "tensor_parallel_size": self.tensor_parallel_degree,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "moe_tensor_parallel_size": self.moe_tensor_parallel_size,
            "moe_expert_parallel_size": self.moe_expert_parallel_size,
            "enable_attention_dp": self.enable_attention_dp,
            "auto_parallel": self.auto_parallel,
            "auto_parallel_world_size": self.auto_parallel_world_size,
            "load_format": self.load_format,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "guided_decoding_backend": self.guided_decoding_backend,
            "iter_stats_max_iterations": self.iter_stats_max_iterations,
            "request_stats_max_iterations": self.request_stats_max_iterations,
            "embedding_parallel_mode": self.embedding_parallel_mode,
            "enable_build_cache": self.enable_build_cache,
            "batching_type": self.batching_type,
            "normalize_log_probs": self.normalize_log_probs,
            "gather_generation_logits": self.gather_generation_logits,
            "max_batch_size": self.max_rolling_batch_size,
            "max_input_len": self.max_input_len,
            "max_seq_len": self.max_seq_len,
            "max_beam_width": self.max_beam_width,
            "max_num_tokens": self.max_num_tokens,
            "backend": self.backend,
            "kv_cache_config": self.get_kv_cache_config(),
            "pytorch_config": self.get_pytorch_config(),
        }
