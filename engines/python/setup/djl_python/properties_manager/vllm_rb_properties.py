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
from enum import Enum
from typing import Optional

from pydantic.v1.class_validators import validator, root_validator

from djl_python.properties_manager.properties import Properties, RollingBatchEnum


class VllmQuantizeMethods(str, Enum):
    awq = 'awq'
    gptq = 'gptq'
    squeezellm = 'squeezellm'


class VllmRbProperties(Properties):
    engine: Optional[str] = None
    dtype: Optional[str] = "auto"
    load_format: Optional[str] = "auto"
    quantize: Optional[VllmQuantizeMethods] = None
    tensor_parallel_degree: Optional[int] = None
    max_rolling_batch_prefill_tokens: Optional[int] = None
    # Adjustable prefix model length for certain 32k or longer model
    max_model_len: Optional[int] = None
    # TODO: change Enforce eager to False once SageMaker driver issue resolved
    enforce_eager: Optional[bool] = None
    # TODO: this default may change with different vLLM versions
    # TODO: try to get good default from vLLM to prevent revisiting
    # TODO: last time check: vllm 0.3.1
    gpu_memory_utilization: Optional[float] = 0.9
    # TODO: speculative decoding changes
    speculative_draft_model: Optional[str] = None
    speculative_length: int = 5
    draft_model_tp_size: int = 1
    record_acceptance_rate: Optional[bool] = False

    @root_validator(skip_on_failure=True)
    def validate_engine(cls, properties):
        engine = properties["engine"]
        rolling_batch = properties["rolling_batch"]
        if rolling_batch == RollingBatchEnum.vllm and engine != "Python":
            raise AssertionError(
                f"Need python engine to start vLLM RollingBatcher")

        if rolling_batch == RollingBatchEnum.lmidist_v2 and engine != "MPI":
            raise AssertionError(
                f"Need MPI engine to start lmidist_v2 RollingBatcher")

        return properties

    # TODO: Remove this once SageMaker resolved driver issue
    @root_validator(skip_on_failure=True)
    def set_eager_model_for_quantize(cls, properties):
        if "quantize" in properties and properties["quantize"] == "awq":
            if properties["enforce_eager"] is None:
                properties["enforce_eager"] = True
        if properties["enforce_eager"] is None:
            properties["enforce_eager"] = False
        return properties
