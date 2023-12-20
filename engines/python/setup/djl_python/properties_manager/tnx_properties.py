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
import os
import re
from typing import Optional, Union, List

from pydantic import validator, root_validator
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


class TnXQuantizeMethods(str, Enum):
    static_int8 = 'static_int8'


TNX_SUPPORTED_ROLLING_BATCH_TYPES = ['auto']


class TransformerNeuronXProperties(Properties):
    """Transformer neuronx related configurations"""
    neuron_optimize_level: Optional[OptimizeLevel] = None
    dtype: Dtype = Dtype.f32
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

    @validator('neuron_optimize_level')
    def set_neuron_optimal_env(cls, level):
        os.environ[
            "NEURON_CC_FLAGS"] = os.environ["NEURON_CC_FLAGS"] + f" -O{level}"

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

    @root_validator()
    def set_amp_value(cls, properties):
        properties['amp'] = properties['dtype'].name
        return properties

    @root_validator()
    def set_quantize(cls, properties):
        if properties['quantize'] and properties[
                'quantize'].value == TnXQuantizeMethods.static_int8.value:
            properties['load_in_8bit'] = True
        return properties
