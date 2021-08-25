#!/usr/bin/env python
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import numpy as np

from util.np_util import djl_to_np_decode


class Preprocessor(object):

    def initialize(self):
        pass

    def preprocess(self, input_data) -> list[np.ndarray]:
        content = input_data.get_content()
        pair_keys = content.get_keys()
        if "data" in pair_keys:
            return content.get_as_numpy("data")
        elif "body" in pair_keys:
            return content.get_as_numpy("body")
        else:
            data = list(content.get_values())[0]
        np_list = djl_to_np_decode(data)
        return np_list
