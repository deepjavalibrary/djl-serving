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
from abc import ABC, abstractmethod


class SearchConfig:

    def __init__(self):
        self.max_new_seqlen = 30
        self._max_seqlen = 0
        self.eos_token_id = 50256
        self.pad_token_id = 50256
        self.topk = 4
        self.alpha = 0.6
        self.beam = 3
        self.sampling = False
        self.topp = 0.92
        self.temperature = 1
