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

import os
import torch
from enum import IntEnum
from typing import Optional, Any
from pydantic import validator, root_validator
from djl_python.properties_manager.properties import Properties


class OptimizeLevel(IntEnum):
    level1 = 1
    level2 = 2
    level3 = 3


class StableDiffusionNeuronXProperties(Properties):
    """Optimum neuronx stable diffusion related configurations"""
    neuron_optimize_level: Optional[OptimizeLevel] = None
    height: int = 512
    width: int = 512
    num_images_per_prompt: Optional[int] = 1
    dtype: Optional[Any] = None
    amp: Optional[str] = 'fp32'
    use_auth_token: Optional[str] = None
    save_mp_checkpoint_path: Optional[str] = None

    @validator('neuron_optimize_level')
    def set_neuron_optimal_env(cls, level):
        os.environ[
            "NEURON_CC_FLAGS"] = os.environ["NEURON_CC_FLAGS"] + f" -O{level}"

    @root_validator()
    def set_dtype(cls, properties):

        def get_torch_dtype_from_str(dtype: str):
            if dtype == "fp32":
                return torch.float32
            if dtype == "bf16":
                return torch.bfloat16
            raise ValueError(
                f"Invalid data type: {dtype}, NeuronX Stable Diffusion only supports fp32, and bf16"
            )

        properties['amp'] = properties.get('dtype', 'fp32')
        properties['dtype'] = get_torch_dtype_from_str(
            properties.get('dtype', 'fp32'))
        return properties
