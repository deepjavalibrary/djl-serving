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

import logging
import os
import re
from typing import Optional, List

from pydantic.v1 import validator, root_validator
from enum import IntEnum, Enum

from djl_python.properties_manager.properties import Properties, RollingBatchEnum, StreamingEnum


class OptimizeLevel(IntEnum):
    level1 = 1
    level2 = 2
    level3 = 3


class Dtype(str, Enum):
    f32 = 'fp32'
    f16 = 'fp16'
    bf16 = 'bf16'


class TnXDtypeName(str, Enum):
    float32 = 'float32'
    float16 = 'float16'
    bfloat16 = 'bfloat16'


class TnXQuantizeMethods(str, Enum):
    static_int8 = 'static_int8'


class TnXGQAMethods(str, Enum):
    shard_over_heads = 'shard-over-heads'
    shard_over_batch = 'shard-over-batch'
    replicated_heads = 'replicated-heads'
    all_gather_heads = 'all-gather-heads'


class TnXModelLoaders(str, Enum):
    tnx = "tnx"
    optimum = "optimum"


class TnXModelSchema(str, Enum):
    legacy = "legacy"
    optimum = "optimum"
    safetensors = "safetensors"
    compile_only = "compile_only"


class TnXGenerationStrategy(str, Enum):
    continuous_batching = "continuous_batching"
    naive_rolling_batch = "naive_rolling_batch"


class TnXMemoryLayout(str, Enum):
    LAYOUT_BSH = "BSH"
    LAYOUT_HSB = "HSB"
    LAYOUT_SBH = "SBH"


TNX_SUPPORTED_ROLLING_BATCH_TYPES = ['auto']


class TransformerNeuronXProperties(Properties):
    """Transformer neuronx related configurations"""
    neuron_optimize_level: Optional[OptimizeLevel] = None
    enable_mixed_precision_accumulation: bool = False
    enable_saturate_infinity: bool = False
    dtype: Dtype = Dtype.f16
    n_positions: int = 128
    unroll: Optional[str] = None
    load_in_8bit: Optional[bool] = False
    low_cpu_mem_usage: bool = False
    load_split_model: bool = False
    context_length_estimate: Optional[List[int]] = None
    amp: Optional[str] = None
    quantize: Optional[TnXQuantizeMethods] = None
    compiled_graph_path: Optional[str] = None
    task: Optional[str] = None
    save_mp_checkpoint_path: Optional[str] = None
    group_query_attention: Optional[str] = None
    model_loader: Optional[TnXModelLoaders] = None
    rolling_batch_strategy: Optional[TnXGenerationStrategy] = None
    fuse_qkv: Optional[bool] = False
    on_device_embedding: Optional[bool] = False
    attention_layout: Optional[TnXMemoryLayout] = None
    collectives_layout: Optional[TnXMemoryLayout] = None
    cache_layout: Optional[TnXMemoryLayout] = None
    partition_schema: Optional[TnXModelSchema] = None
    all_reduce_dtype: Optional[TnXDtypeName] = None
    cast_logits_dtype: Optional[TnXDtypeName] = None

    @validator('neuron_optimize_level')
    def set_neuron_optimal_env(cls, level):
        if "NEURON_CC_FLAGS" not in os.environ:
            os.environ["NEURON_CC_FLAGS"] = ""
        os.environ[
            "NEURON_CC_FLAGS"] = os.environ["NEURON_CC_FLAGS"] + f" -O{level}"

    @validator('enable_mixed_precision_accumulation')
    def set_mixed_precision_accumulation(cls, enablement):
        if "NEURON_CC_FLAGS" not in os.environ:
            os.environ["NEURON_CC_FLAGS"] = ""
        os.environ["NEURON_CC_FLAGS"] = os.environ[
            "NEURON_CC_FLAGS"] + f" --enable-mixed-precision-accumulation"

    @validator('enable_saturate_infinity')
    def set_saturate_infinity(cls, enablement):
        if "NEURON_CC_FLAGS" not in os.environ:
            os.environ["NEURON_CC_FLAGS"] = ""
        os.environ["NEURON_CC_FLAGS"] = os.environ[
            "NEURON_CC_FLAGS"] + f" --enable-saturate-infinity"

    @validator('context_length_estimate', pre=True)
    def parse_context_length(cls, context_length_estimate):
        return [
            int(context_length)
            for context_length in context_length_estimate.split(',')
        ]

    @validator('rolling_batch', pre=True)
    def validate_rolling_batch(cls, rolling_batch: str) -> str:
        if rolling_batch == RollingBatchEnum.disable.value:
            return rolling_batch
        if rolling_batch not in TNX_SUPPORTED_ROLLING_BATCH_TYPES:
            logging.warning(
                f"transformer neuronx only supports "
                f"rolling batch type {TNX_SUPPORTED_ROLLING_BATCH_TYPES}."
                f"choosing neuronx rolling batch automatically.")
            return 'auto'
        return rolling_batch

    @validator('batch_size')
    def validate_batch_size(cls, batch_size: int, values) -> int:
        """
        Transformer neuronx supports batch_size, and max_rolling_batch_size.
        The batch_size param is for dynamic batching, and max_rolling_batch_size is for rolling batch.
        We validate here that the values are compatible and set for both compilation and inference.
        """
        if batch_size > 1:
            if values['rolling_batch'] != RollingBatchEnum.disable:
                raise ValueError(
                    "Dynamic batching and rolling batch cannot be enabled at the same time, please "
                    "set either batch size or rolling batch with max_rolling_batch_size, but not both."
                )
        return batch_size

    @validator('compiled_graph_path')
    def validate_compiled_graph_path(cls, path: str) -> str:
        """Transformer neuronx accepts compiled graph paths as directories and s3 uri"""
        if not re.search("^s3:\/\/([^/]+)\/([\w\W]+)", path):
            if not os.path.isdir(path):
                raise ValueError(
                    f"{path} is not a valid value for compiled_graph_path. "
                    f"Supported values are: directories, and S3 URIs to directories."
                )
            else:
                path = os.path.join(os.getcwd(), path)
        os.environ["NEURON_COMPILE_CACHE_URL"] = path
        return path

    @validator('group_query_attention')
    def validate_gqa(cls, gqa: str) -> str:
        """
        Transformers neuronx supports GQA for Llama and Mistral model variants.
        We validate here that the value provided maps to a known GQA type support by transformers neuronx
        """
        try:
            return TnXGQAMethods(gqa).value
        except ValueError:
            raise ValueError(
                f"{gqa} is not a valid value for group_query_attention. "
                f"Supported values are: {[v.value for v in TnXGQAMethods]}")

    @root_validator(skip_on_failure=True)
    def set_amp_value(cls, properties):
        properties['amp'] = properties['dtype'].name
        return properties

    @root_validator(skip_on_failure=True)
    def set_quantize(cls, properties):
        if properties['quantize'] and properties[
                'quantize'].value == TnXQuantizeMethods.static_int8.value:
            properties['load_in_8bit'] = True
        return properties

    @root_validator(skip_on_failure=True)
    def validate_schema_loader_combination(cls, properties):
        if properties.get('model_loader') and properties[
                'model_loader'].value == TnXModelLoaders.tnx.value:
            if properties.get('partition_schema') and properties[
                    'partition_schema'] == TnXModelSchema.optimum:
                raise ValueError(
                    f"Transformers NeuronX model loader does not support optimum cache partitioning. "
                    f"Supported values are: {[v.value for v in TnXModelSchema if v.value != TnXModelSchema.optimum]}"
                )
        if properties.get('model_loader') and properties[
                'model_loader'].value == TnXModelLoaders.optimum.value:
            if properties.get('partition_schema') and properties[
                    'partition_schema'] != TnXModelSchema.optimum:
                raise ValueError(
                    f"Optimum model loader does not support non-optimum cache partitioning. "
                    f"Supported values are: {TnXModelSchema.optimum.value}")

        if properties.get('partition_schema') and properties[
                'partition_schema'] == TnXModelSchema.optimum and properties.get(
                    'model_loader') is None:
            properties['model_loader'] = TnXModelLoaders.optimum
        if properties.get('partition_schema') and properties[
                'partition_schema'] != TnXModelSchema.optimum and properties.get(
                    'model_loader') is None:
            properties['model_loader'] = TnXModelLoaders.tnx
        return properties

    @root_validator(pre=True)
    def set_model_loader(cls, properties):
        if properties.get('model_loader') is None:
            if properties.get('fuse_qkv') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('collectives_layout') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('on_device_embedding') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('load_in_8bit') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('quantize') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('load_split_model') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('compiled_graph_path') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
            elif properties.get('context_length_estimate') is not None:
                properties['model_loader'] = TnXModelLoaders.tnx
        return properties
