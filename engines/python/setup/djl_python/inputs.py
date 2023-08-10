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

import io
import struct
import json
import re

from .np_util import from_nd_list
from .pair_list import PairList


def retrieve_buffer(conn, length):
    """
    Retrieves buffer in the specified length.
    :param conn: socket connection
    :param length: length of the data to be read
    :return: retrieved byte array
    """
    data = bytearray()

    while length > 0:
        pkt = conn.recv(length)
        if len(pkt) == 0:
            raise ValueError("Connection disconnected")

        data += pkt
        length -= len(pkt)

    return data


def retrieve_int(conn):
    """
    Retrieves int value.
    :param conn: socket connection
    :return: retrieved integer value
    """
    data = retrieve_buffer(conn, 4)
    return struct.unpack(">i", data)[0]


def retrieve_short(conn):
    """
    Retrieves int value.
    :param conn: socket connection
    :return: retrieved integer value
    """
    data = retrieve_buffer(conn, 2)
    return struct.unpack(">h", data)[0]


def retrieve_utf8(conn):
    length = retrieve_short(conn)
    if length < 0:
        return None

    data = retrieve_buffer(conn, length)
    return data.decode("utf8")


class Input(object):

    def __init__(self):
        self.function_name = None
        self.properties = dict()
        self.content = PairList()

    def __str__(self):
        cur_str = "properties: " + str(self.get_properties())
        for key in self.get_content().get_keys():
            cur_str += "\n{}: {}".format(key, self.get_data(key))
        return cur_str

    def is_batch(self) -> bool:
        return self.get_batch_size() > 1

    def get_batch_size(self) -> int:
        return int(self.properties.get("batch_size", "1"))

    def get_batches(self) -> list:
        batch_size = self.get_batch_size()
        batch = []
        for i in range(batch_size):
            item = Input()
            prefix = f"batch_{i}."
            length = len(prefix)
            for key, value in self.properties.items():
                if key.startswith(prefix):
                    key = key[length:]
                    item.properties[key] = value
                elif not key.startswith("batch_"):
                    item.properties[key] = value

            batch.append(item)

        p = re.compile("batch_(\\d+)\\.(.*)")
        for i in range(self.content.size()):
            key = self.content.key_at(i)
            m = p.match(key)
            if m is None:
                raise ValueError(f"Unexpected key in batch input: {key}")
            index = int(m.group(1))
            batch[index].content.add(m.group(2), self.content.value_at(i))

        return batch

    def get_function_name(self) -> str:
        return self.function_name

    def get_properties(self) -> dict:
        """
        Returns the properties

        :return: properties
        """
        return self.properties

    def get_content(self) -> PairList:
        """
        Returns the content

        :return: content
        """
        return self.content

    def get_property(self, key: str) -> str:
        """
        Returns the value of a property key

        :param key: key of map
        :return: value of the key
        """
        return next(
            (v
             for k, v in self.properties.items() if k.lower() == key.lower()),
            None)

    def contains_key(self, key) -> bool:
        return self.content.get(key) is not None

    def get_data(self, key=None):
        content_type = self.get_property("content-type")
        if content_type == "tensor/ndlist":
            return self.get_as_numpy(key)
        if content_type == "tensor/npz":
            return self.get_as_npz(key)
        elif content_type is not None and content_type.startswith(
                "application/json"):
            return self.get_as_json(key)
        elif content_type is not None and content_type.startswith("text/"):
            return self.get_as_string(key)
        elif content_type is not None and content_type.startswith("image/"):
            return self.get_as_image(key)
        elif self.content.is_empty():
            return None
        else:
            return self.get_as_bytes(key=key)

    def get_as_bytes(self, key=None):
        if self.content.is_empty():
            return None

        if key:
            ret = self.content.get(key)
            if ret is None:
                raise KeyError(
                    "requested data for key:{} is not found!".format(key))
            return ret

        ret = self.content.get("data")

        if ret is None:
            ret = self.content.value_at(0)
        return ret

    def get_as_string(self, key=None) -> str:
        return self.get_as_bytes(key=key).decode("utf-8")

    def get_as_json(self, key=None):
        return json.loads((self.get_as_bytes(key=key).decode("utf-8")))

    def get_as_image(self, key=None):
        from PIL import Image
        return Image.open(io.BytesIO(self.get_as_bytes(key=key)))

    def get_as_numpy(self, key=None) -> list:
        """
        Returns
            1. value as numpy list if key is provided
            2. list of values as numpy list if key is not provided
        :param key: optional key
        :return: list of numpy array
        """
        return from_nd_list(self.get_as_bytes(key=key))

    def get_as_npz(self, key=None) -> list:
        import numpy
        npz = numpy.load(io.BytesIO(self.get_as_bytes(key=key)))
        result = [npz[name] for name in npz.files]
        return result

    def get_as_csv(self, key=None) -> list:
        import csv
        stream = io.StringIO(self.get_as_string(key=key))
        return list(csv.DictReader(stream))

    def is_empty(self):
        return self.content.is_empty()

    def read(self, conn):
        prop_size = retrieve_short(conn)

        for _ in range(prop_size):
            key = retrieve_utf8(conn)
            val = retrieve_utf8(conn)
            self.properties[key] = val

        content_size = retrieve_short(conn)

        for _ in range(content_size):
            key = retrieve_utf8(conn)
            length = retrieve_int(conn)
            val = retrieve_buffer(conn, length)
            self.content.add(key=key, value=val)

        self.function_name = self.properties.get('handler')
