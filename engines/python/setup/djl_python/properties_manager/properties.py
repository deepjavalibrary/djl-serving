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
from typing import Optional

from pydantic import BaseModel, model_validator, field_validator, Field


class RollingBatchEnum(str, Enum):
    auto = "auto"
    disable = "disable"


class StreamingEnum(str, Enum):
    true = "true"
    false = "false"
    huggingface = "huggingface"


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
    enable_streaming: StreamingEnum = StreamingEnum.true
    batch_size: int = 1
    max_rolling_batch_size: int = 32
    dtype: Optional[str] = None
    revision: Optional[str] = None

    @field_validator('enable_streaming', mode='before')
    def validate_enable_streaming(cls, enable_streaming: str) -> str:
        return enable_streaming.lower()

    @field_validator('batch_size')
    def validate_batch_size(cls, batch_size: int, fields) -> int:
        properties = fields.data
        if batch_size > 1:
            if properties[
                    'rolling_batch'] == RollingBatchEnum.disable and properties[
                        'enable_streaming'] != "false":
                raise ValueError(
                    "We cannot enable streaming for dynamic batching")
        return batch_size

    @model_validator(mode='before')
    def set_model_id_or_path(cls, properties: dict) -> dict:
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise, we assume model artifacts are in the model_dir
        properties['model_id_or_path'] = properties.get(
            "model_id") or properties.get("model_dir")
        return properties
