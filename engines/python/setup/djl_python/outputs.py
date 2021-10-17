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

import json
import struct

from .np_util import to_nd_list
from .pair_list import PairList


class Output(object):
    def __init__(self):
        self.code = 200
        self.message = 'OK'
        self.properties = dict()
        self.content = PairList()

    def __str__(self):
        d = dict()
        for i in range(self.content.size()):
            v = "type: " + str(type(self.content.value_at(i)))
            d[self.content.key_at(i)] = v
        return json.dumps(
            {
                "code": self.code,
                "message": self.message,
                "properties": self.properties,
                "content": d
            },
            indent=2)

    def set_code(self, code):
        self.code = code

    def set_message(self, message):
        self.message = message

    def add_property(self, key, val):
        self.properties[key] = val

    def add(self, value, key=None):
        if value is str:
            self.add(key=key, value=value.encode("utf-8"))
        elif value is bytearray:
            self.add(key=key, value=value)
        elif value is bytes:
            self.add(key=key, value=bytearray(value))
        else:
            self.add_as_json(value, key=key)

    def add_as_numpy(self, np_list, key=None):
        self.content.add(key=key, value=to_nd_list(np_list))

    def add_as_json(self, val, key=None):
        json_value = json.dumps(val, indent=2).encode("utf-8")
        self.content.add(key=key, value=json_value)

    @staticmethod
    def write_utf8(msg, val):
        if val is None:
            msg += struct.pack('>h', -1)
        else:
            buf = val.encode('utf-8')
            msg += struct.pack('>h', len(buf))
            msg += buf

    def encode(self) -> bytearray:
        msg = bytearray()
        msg += struct.pack('>h', self.code)
        self.write_utf8(msg, self.message)

        msg += struct.pack('>h', len(self.properties))
        for k, v in self.properties.items():
            self.write_utf8(msg, k)
            self.write_utf8(msg, v)

        size = self.content.size()
        msg += struct.pack('>h', size)
        for i in range(size):
            k = self.content.key_at(i)
            v = self.content.value_at(i)
            self.write_utf8(msg, k)
            msg += struct.pack('>i', len(v))
            msg += v

        return msg
