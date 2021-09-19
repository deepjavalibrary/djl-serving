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

import struct
import numpy as np

from np_util import from_nd_list
from pair_list import PairList


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
        self.request_id = None
        self.properties = dict()
        self.content = PairList()

    def get_function_name(self) -> str:
        return self.function_name

    def get_request_id(self) -> str:
        """
        Returns the request id

        :return: request_id
        """
        return self.request_id

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

    def get_properties_value(self, key: str) -> str:
        """
        Returns the value of a property key

        :param key: key of map
        :return: value of the key
        """
        return self.properties[key]

    def get_as_numpy(self, key=None) -> list[np.ndarray]:
        """
        Returns
            1. value as numpy list if key is provided
            2. list of values as numpy list if key is not provided
        :param key: optional key
        :return: list of numpy array
        """
        # return list of values as numpy list if not provided key
        if key is None:
            values = self.content.get_values()
            result = []
            for value in values:
                result.extend(from_nd_list(value))
            return result
        else:
            value = self.content.get(key=key)
            return from_nd_list(value)

    def read(self, conn):
        self.request_id = retrieve_utf8(conn)
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
