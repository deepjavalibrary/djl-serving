#!/usr/bin/env python
#
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


class PairList(object):
    def __init__(self, keys=None, values=None, pair_list=None, pair_map=None):
        if keys and values:
            if len(keys) != len(values):
                raise ValueError("key value size mismatch.")
            self.keys = keys
            self.values = values
        elif pair_list:
            for pair in pair_list:
                self.keys.append(pair[0])
                self.values.append(pair[1])
        elif pair_map:
            for key, value in pair_map.items():
                self.keys.append(key)
                self.values.append(value)
        else:
            self.keys = []
            self.values = []

    def add(self, key=None, value=None, index=None, pair=None):
        if index and value:
            self.keys.insert(index, key)
            self.values.insert(index, value)
        elif pair:
            self.keys.append(pair[0])
            self.values.append(pair[1])
        elif value:
            self.keys.append(key)
            self.values.append(value)

    def add_all(self, other):
        if other:
            self.keys.extend(other.keys())
            self.values.extend(other.values())

    def size(self):
        return len(self.keys)

    def is_empty(self):
        return self.size() == 0

    def get(self, index=None, key=None):
        if index:
            return self.keys[0], self.values[index]
        elif key:
            if key not in self.keys:
                return None
            key_index = self.keys.index(key)
            return self.values[key_index]

    def key_at(self, index: int):
        return self.keys[index]

    def value_at(self, index: int):
        return self.values[index]

    def get_keys(self):
        return self.keys

    def get_values(self) -> list:
        return self.values
