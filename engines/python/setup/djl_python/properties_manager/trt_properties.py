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
from djl_python.properties_manager.properties import Properties, RollingBatchEnum
from pydantic import field_validator

TRT_SUPPORTED_ROLLING_BATCH_TYPES = [
    RollingBatchEnum.auto.value, RollingBatchEnum.trtllm.value,
    RollingBatchEnum.disable.value
]


class TensorRtLlmProperties(Properties):

    @field_validator('rolling_batch', mode='before')
    def validate_rolling_batch(cls, rolling_batch: str) -> str:
        rolling_batch = rolling_batch.lower()

        if rolling_batch not in TRT_SUPPORTED_ROLLING_BATCH_TYPES:
            raise ValueError(
                f"tensorrt llm only supports "
                f"rolling batch type {TRT_SUPPORTED_ROLLING_BATCH_TYPES}.")

        return rolling_batch
