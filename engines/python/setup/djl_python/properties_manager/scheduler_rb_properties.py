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

from djl_python.properties_manager.hf_properties import HuggingFaceProperties

DEFAULT_SEARCH_ALGORITHM = 'greedy'


class SchedulerRbProperties(HuggingFaceProperties):
    decoding_strategy: Optional[str] = DEFAULT_SEARCH_ALGORITHM
    disable_flash_attn: Optional[bool] = True
    # a threshold to limit the max padding sparsity
    max_sparsity: Optional[float] = 0.33
    # a threshold to limit the max number of batch splits
    max_splits: Optional[int] = 3
    # TODO: Deprecated, remove this implementation
    multi_gpu: Optional[str] = None
