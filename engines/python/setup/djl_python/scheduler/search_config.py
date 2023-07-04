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

    def __init__(self, **kwargs):
        self.max_new_seqlen = kwargs.get('max_new_tokens', 30)
        self._max_seqlen = 0
        self.eos_token_id = kwargs.get('eos_token_id', 50256)
        self.pad_token_id = kwargs.get('pad_token_id', 50256)
        self.topk = kwargs.get('top_k', 4)
        self.alpha = kwargs.get('penalty_alpha', 0.6)
        self.beam = kwargs.get('num_beams', 3)
        self.sampling = kwargs.get('do_sample', False)
        self.topp = kwargs.get('top_p', 0.92)
        self.temperature = kwargs.get('temperature', 1)
        self.use_lru_kv_cache = kwargs.get('use_lru_kv_cache', False)
