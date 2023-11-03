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
from typing import Optional

from pydantic import field_validator, model_validator
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
    bitsandbytes8 = 'bitsandbytes8'


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
    context_length_estimate: Optional[dict] = None
    amp: Optional[str] = None
    quantize: Optional[TnXQuantizeMethods] = None
    compiled_graph_path: Optional[str] = None

    @field_validator('neuron_optimize_level')
    def set_neuron_optimal_env(cls, level):
        os.environ[
            "NEURON_CC_FLAGS"] = os.environ["NEURON_CC_FLAGS"] + f" -O{level}"

    @field_validator('context_length_estimate', mode='before')
    def parse_context_length(cls, context_length_estimate):
        return json.loads(context_length_estimate)

    @field_validator('rolling_batch', mode='before')
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

    @field_validator('batch_size')
    def validate_batch_size(cls, batch_size: int, fields) -> int:
        """
        Transformer neuronx has both option.batch_size and batch_size.
        option.batch_size is to compile the model with batch size, which cannot be
        differentiated in neuronx handlers. Hence, just throwing a warning here.
        """
        properties = fields.data
        if batch_size > 1:
            if properties[
                    'rolling_batch'] == RollingBatchEnum.disable and properties[
                        'enable_streaming'] != StreamingEnum.false:
                logging.warning(
                    "We cannot enable streaming for dynamic batching")
        return batch_size

    @field_validator('compiled_graph_path')
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

    @model_validator(mode='after')
    def set_amp_value(self, validation_info):
        self.amp = self.dtype.name
        return self

    @model_validator(mode='after')
    def set_quantize(self, validation_info):
        if self.quantize and self.quantize.value == TnXQuantizeMethods.bitsandbytes8.value:
            self.load_in_8bit = True
        return self
