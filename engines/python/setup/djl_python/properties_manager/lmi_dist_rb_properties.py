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

from typing import Optional

from pydantic.class_validators import validator

from engines.python.setup.djl_python.properties_manager.properties import Properties


class LmiDistRbProperties(Properties):
    engine : Optional[str] = None
    quantize: Optional[str] = None
    tensor_parallel_degree: Optional[int] = -1
    paged_attention: Optional[bool] = True
    max_rolling_batch_prefill_tokens: Optional[int] = -1

    @validator('engine')
    def validate_engine(cls, engine):
        if engine != "MPI":
            raise AssertionError(
                f"Need MPI engine to start lmi-dist RollingBatcher")
        return engine
