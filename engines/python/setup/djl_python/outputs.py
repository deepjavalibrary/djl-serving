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
import logging

from .np_util import to_nd_list
from .pair_list import PairList


# https://github.com/automl/SMAC3/issues/453
class _JSONEncoder(json.JSONEncoder):
    """
    custom `JSONEncoder` to make sure float and int64 ar converted
    """

    def default(self, obj):
        import datetime
        if isinstance(obj, datetime.datetime):
            return obj.__str__()

        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        return super(_JSONEncoder, self).default(obj)


class Output(object):

    def __init__(self, code=200, message='OK'):
        self.code = code
        self.message = message
        self.properties = dict()
        self.content = PairList()
        self.stream_content = None

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

    def set_code(self, code: int):
        self.code = code
        return self

    def set_message(self, message: str):
        self.message = message
        return self

    def error(self, error: str, code=424, message="prediction failure"):
        self.code = code
        self.message = message
        body = {"code": self.code, "message": self.message, "error": error}
        self.add_property("content-type", "application/json")
        self.add_as_json(body)
        return self

    def add_property(self, key, val):
        self.properties[key] = val
        return self

    def add(self, value, key=None, batch_index=None):
        if key is not None and type(key) is not str:
            logging.warning(f"Output key should be str type, got {type(key)}")
            key = str(key)

        if batch_index is not None:
            key = "" if key is None else key
            key = f"batch_{batch_index}.{key}"

        if type(value) is str:
            self.content.add(key=key, value=value.encode("utf-8"))
        elif type(value) is bytearray:
            self.content.add(key=key, value=value)
        elif type(value) is bytes:
            self.content.add(key=key, value=bytearray(value))
        else:
            self.content.add(key=key, value=self._encode_json(value))
        return self

    def add_as_numpy(self, np_list, key=None, batch_index=None):
        return self.add(to_nd_list(np_list), key=key, batch_index=batch_index)

    def add_as_npz(self, np_list, key=None, batch_index=None):
        import numpy as np
        import io
        memory_file = io.BytesIO()
        np.savez(memory_file, *np_list)
        memory_file.seek(0)
        return self.add(memory_file.read(-1), key=key, batch_index=batch_index)

    def add_as_json(self, val, key=None, batch_index=None):
        return self.add(self._encode_json(val),
                        key=key,
                        batch_index=batch_index)

    def add_stream_content(self, stream_content):
        self.stream_content = stream_content

    @staticmethod
    def _encode_json(val) -> bytes:
        return bytearray(
            json.dumps(val,
                       ensure_ascii=False,
                       allow_nan=False,
                       indent=2,
                       cls=_JSONEncoder,
                       separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def write_utf8(msg, val):
        if val is None:
            msg += struct.pack('>h', -1)
        else:
            buf = val.encode('utf-8')
            msg += struct.pack('>h', len(buf))
            msg += buf

    def send(self, cl_socket):
        msg = bytearray()
        msg += struct.pack('>h', self.code)
        self.write_utf8(msg, self.message)

        msg += struct.pack('>h', len(self.properties))
        for k, v in self.properties.items():
            self.write_utf8(msg, k)
            self.write_utf8(msg, v)

        if self.stream_content is None:
            size = self.content.size()
            msg += struct.pack('>h', size)
            for i in range(size):
                k = self.content.key_at(i)
                v = self.content.value_at(i)
                self.write_utf8(msg, k)
                msg += struct.pack('>i', len(v))
                msg += v
            cl_socket.sendall(msg)
            return

        msg += struct.pack('>h', -1)
        cl_socket.sendall(msg)

        while True:
            try:
                data = next(self.stream_content)

                msg = bytearray()
                msg += b'\1'

                if type(data) is str:
                    data = data.encode('utf-8')
                elif type(data) is bytearray:
                    pass
                elif type(data) is bytes:
                    data = bytearray(data)
                else:
                    data = bytearray(self._encode_json(data))

                msg += struct.pack('>i', len(data))
                msg += data
                cl_socket.sendall(msg)
            except StopIteration:
                msg = bytearray()
                msg += b'\0'
                msg += struct.pack('>i', 0)
                cl_socket.sendall(msg)
                break
            except Exception as e:
                logging.exception("Failed read streaming content from output")
                msg = bytearray()
                msg += b'\0'
                data = str(e).encode('utf-8')
                msg += struct.pack('>i', len(data))
                msg += data
                cl_socket.sendall(msg)
                break
