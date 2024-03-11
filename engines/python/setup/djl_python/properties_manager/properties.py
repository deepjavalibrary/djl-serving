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
from enum import Enum
from typing import Optional, Union, Callable
from pydantic.v1 import BaseModel, root_validator, validator, Field


class RollingBatchEnum(str, Enum):
    vllm = "vllm"
    lmidist = "lmi-dist"
    ## this is temporary and will replace lmidist after testing
    lmidist_v2 = "lmi-dist-v2"
    scheduler = "scheduler"
    auto = "auto"
    disable = "disable"
    trtllm = "trtllm"
    deepspeed = "deepspeed"


class StreamingEnum(str, Enum):
    true = "true"
    false = "false"
    huggingface = "huggingface"


def is_streaming_enabled(enable_streaming: StreamingEnum) -> bool:
    return enable_streaming.value != StreamingEnum.false.value


def is_rolling_batch_enabled(rolling_batch: RollingBatchEnum) -> bool:
    return rolling_batch.value != RollingBatchEnum.disable.value


class Properties(BaseModel):
    """ Configures common properties for all engines """
    # Required configurations from user
    model_id_or_path: str
    # Optional configurations with default values
    # Make the default to auto, after java front end changes and test cases are changed.
    rolling_batch: RollingBatchEnum = RollingBatchEnum.disable
    tensor_parallel_degree: int = 1
    trust_remote_code: bool = os.environ.get("HF_TRUST_REMOTE_CODE",
                                             "FALSE").lower() == 'true'
    # TODO: disabling streaming, as it is not supported for all models of the frameworks. Will revisit this
    enable_streaming: StreamingEnum = StreamingEnum.false
    batch_size: int = 1
    max_rolling_batch_size: Optional[int] = 32
    dtype: Optional[str] = None
    revision: Optional[str] = None
    output_formatter: Optional[Union[str, Callable]] = None
    waiting_steps: Optional[int] = None
    is_mpi: bool = False

    # Spec_dec
    draft_model_id: Optional[str] = None
    spec_length: Optional[int] = 0

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def calculate_is_mpi(cls, properties):
        properties['is_mpi'] = properties.get("mpi_mode") == "true"
        return properties

    @validator('enable_streaming', pre=True)
    def validate_enable_streaming(cls, enable_streaming: str) -> str:
        return enable_streaming.lower()

    @validator('batch_size', pre=True)
    def validate_batch_size(cls, batch_size, values):
        batch_size = int(batch_size)
        if batch_size > 1:
            if not is_rolling_batch_enabled(
                    values['rolling_batch']) and is_streaming_enabled(
                        values['enable_streaming']):
                raise ValueError(
                    "We cannot enable streaming for dynamic batching")
        return batch_size

    @root_validator(pre=True)
    def set_model_id_or_path(cls, properties: dict) -> dict:
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise, we assume model artifacts are in the model_dir
        properties['model_id_or_path'] = properties.get(
            "model_id") or properties.get("model_dir")
        return properties
