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

"""
Contains util functions for encoding and decoding of data.
"""
import struct

from protocol.input import Input
from protocol.output import Output
from util.pair_list import PairList


def set_int(value: int) -> bytes:
    """
    Converts int value into bytes.
    :param value: integer
    :return: converted bytes
    """
    return struct.pack(">i", value)


def set_long(value: int) -> bytes:
    return struct.pack(">q", value)


def set_str(value: str) -> bytes:
    return struct.pack(">h", len(value)) + bytes(value, "utf8")


def set_char(value: str) -> bytes:
    return struct.pack('>h', ord(value))


def construct_enc_response(arr: bytearray) -> bytearray:
    """
    Constructs the response to be sent. length of the array + data.
    :param arr: bytearray
    :return: response format
    """
    response = bytearray()
    response.extend(set_int(len(arr)))
    response.extend(arr)
    return response


def get_byte_as_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    return encoded[idx], idx + 1


def get_bytes(encoded: bytearray, idx: int, length: int) -> tuple[bytes, int]:
    return encoded[idx:idx + length], idx + length


def get_char(encoded: bytearray, idx: int) -> tuple[str, int]:
    chr_size = 2
    return chr(struct.unpack(">h", encoded[idx:idx + chr_size])[0]), idx + chr_size


def get_str(encoded: bytearray, idx: int) -> tuple[str, int]:
    length = struct.unpack(">h", encoded[idx:idx + 2])[0]
    idx += 2
    return encoded[idx:idx + length].decode("utf8"), idx + length


def get_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    int_size = 4
    return struct.unpack(">i", encoded[idx:idx + int_size])[0], idx + int_size


def get_long(encoded: bytearray, idx: int) -> tuple[int, int]:
    long_size = 8
    return struct.unpack(">q", encoded[idx:idx + long_size])[0], idx + long_size


def decode_input(arr: bytearray) -> Input:
    idx = 0
    _input = Input()
    req_id, idx = get_str(arr, idx)
    _input.set_request_id(req_id)
    prop_size, idx = get_int(arr, idx)

    for _ in range(prop_size):
        key, idx = get_str(arr, idx)
        val, idx = get_str(arr, idx)
        _input.add_property(key, val)

    content_size, idx = get_int(arr, idx)
    keys = []
    for _ in range(content_size):
        key, idx = get_str(arr, idx)
        keys.append(key)

    values = []
    for _ in range(content_size):
        val_len, idx = get_int(arr, idx)
        val, idx = get_bytes(arr, idx, val_len)
        values.append(val)

    content = PairList(keys=keys, values=values)
    _input.set_content(content)
    return _input
