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
from enum import Enum
from typing import Optional

import torch
from pydantic.class_validators import root_validator

from djl_python.properties_manager.hf_properties import get_torch_dtype_from_str
from djl_python.properties_manager.properties import Properties


class LmiDistQuantizeMethods(str, Enum):
    # added for backward compatibility lmi-dist
    bitsandbytes = 'bitsandbytes'
    bitsandbytes8 = 'bitsandbytes8'
    gptq = 'gptq'
    awq = 'awq'


class LmiDistRbProperties(Properties):
    engine: Optional[str] = None
    quantize: Optional[LmiDistQuantizeMethods] = None
    tensor_parallel_degree: Optional[int] = 1
    paged_attention: Optional[bool] = True
    max_rolling_batch_prefill_tokens: Optional[int] = 4096
    device: Optional[int] = None
    dtype: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None

    @root_validator()
    def validate_mpi_mode(cls, properties):
        if not properties.get("is_mpi") and int(
                properties.get("tensor_parallel_degree", "1")) != 1:
            raise ValueError(f"Need mpi_mode to start lmi-dist RollingBatcher."
                             f"Try with engine=MPI in your serving.properties")
        return properties

    @root_validator()
    def validate_quantize(cls, properties):
        if properties.get('quantize') is None:
            if properties.get('dtype') == "int8":
                properties['quantize'] = LmiDistQuantizeMethods.bitsandbytes
        else:
            # parsing bitsandbytes8, so it can be directly passed to lmi dist model loader.
            if properties.get(
                    'quantize') == LmiDistQuantizeMethods.bitsandbytes8:
                properties['quantize'] = LmiDistQuantizeMethods.bitsandbytes
            if properties.get('dtype') is not None:
                raise ValueError(
                    f"Can't set both dtype: {properties['dtype']} and quantize: {properties['quantize']}"
                )
        return properties

    @root_validator()
    def set_device(cls, properties):
        if properties.get('is_mpi'):
            properties['device'] = int(os.getenv("LOCAL_RANK", 0))
        return properties

    @root_validator()
    def construct_dtype(cls, properties):
        if properties.get('dtype'):
            properties["torch_dtype"] = get_torch_dtype_from_str(
                properties['dtype'].lower())
        elif properties.get('data_type'):
            logging.warning('option.data_type is deprecated.'
                            'Please use option.dtype')
            properties["torch_dtype"] = get_torch_dtype_from_str(
                properties['data_type'].lower())
        else:
            properties['torch_dtype'] = torch.float16
        return properties
