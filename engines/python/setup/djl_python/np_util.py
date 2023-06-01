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

import numpy as np

MAGIC_NUMBER = "NDAR"
VERSION = 3


def set_int(value: int) -> bytes:
    """
    Converts int value into bytes.
    :param value: integer
    :return: bytes
    """
    return struct.pack(">i", value)


def set_long(value: int) -> bytes:
    """
    Converts long value into bytes
    :param value: long
    :return: bytes
    """
    return struct.pack(">q", value)


def set_str(value: str) -> bytes:
    """
    Converts string value into bytes
    :param value: string
    :return: bytes
    """
    return struct.pack(">h", len(value)) + bytes(value, "utf8")


def set_char(value: str) -> bytes:
    """
    Converts char value into bytes
    :param value: char
    :return: bytes
    """
    return struct.pack('>h', ord(value))


def get_byte_as_int(encoded: bytearray, idx: int) -> tuple:
    """
    Returns a byte value as int and next to be read index
    :param encoded: bytearray
    :param idx: index to lookup
    :return: tuple of byte read and next index
    """
    return encoded[idx], idx + 1


def get_bytes(encoded: bytearray, idx: int, length: int) -> tuple:
    """
    Returns the bytes with given length, and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :param length: length to be read
    :return: tuple of bytes read and next index
    """
    return encoded[idx:idx + length], idx + length


def get_char(encoded: bytearray, idx: int) -> tuple:
    """
    Returns a char and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of char and next index
    """
    return chr(struct.unpack(">h", encoded[idx:idx + 2])[0]), idx + 2


def get_str(encoded: bytearray, idx: int) -> tuple:
    """
    Returns a string and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of string and next index
    """
    length = struct.unpack(">h", encoded[idx:idx + 2])[0]
    idx += 2
    return encoded[idx:idx + length].decode("utf8"), idx + length


def get_int(encoded: bytearray, idx: int) -> tuple:
    """
    Returns an integer and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of int and next index
    """
    return struct.unpack(">i", encoded[idx:idx + 4])[0], idx + 4


def get_long(encoded: bytearray, idx: int) -> tuple:
    """
    Returns the long value and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of long and next index
    """
    long_size = 8
    return struct.unpack(">q",
                         encoded[idx:idx + long_size])[0], idx + long_size


def from_nd_list(encoded: bytearray) -> list:
    """
    Converts djl format to list of numpy array
    :param encoded: bytearray
    :return: list of numpy array
    """
    if len(encoded) >= 4 and encoded[0] == 80 and encoded[1] == 75:
        # Assume the input is npz format (PK)
        result = []
        npz = np.load(io.BytesIO(encoded))
        for item in npz.items():
            result.append(item[1])
        return result

    idx = 0
    num_ele, idx = get_int(encoded, idx)
    result = []
    for _ in range(num_ele):
        magic, idx = get_str(encoded, idx)
        if magic != MAGIC_NUMBER:
            raise AssertionError("magic number is not NDAR, actual " + magic)
        version, idx = get_int(encoded, idx)
        if version != VERSION:
            raise AssertionError(f"require version {VERSION}, actual " + str(version))
        flag, idx = get_byte_as_int(encoded, idx)
        if flag == 1:
            _, idx = get_str(encoded, idx)
        _, idx = get_str(encoded, idx)  # ignore sparse format
        datatype, idx = get_str(encoded, idx)
        shape, idx = _shape_decode(encoded, idx)
        order, idx = get_byte_as_int(encoded, idx)
        data_length, idx = get_int(encoded, idx)
        data, idx = get_bytes(encoded, idx, data_length)
        nd = np.ndarray(shape, np.dtype(datatype.lower()), data)
        nd = nd.newbyteorder(chr(order))
        result.append(nd)
    return result


def to_nd_list(np_list) -> bytearray:
    """
    Converts list of numpy array into djl NDList

    :param np_list: list of numpy array
    :return: djl NDList as bytearray
    """
    arr = bytearray()
    if type(np_list) is not list:
        np_list = [np_list]

    arr.extend(set_int(len(np_list)))
    for nd in np_list:
        arr.extend(set_str(MAGIC_NUMBER))
        arr.extend(set_int(VERSION))
        arr.append(0)  # no name
        arr.extend(set_str("default"))
        arr.extend(set_str(str(nd.dtype).upper()))
        _shape_encode(nd.shape, arr)
        arr.append(ord('<'))  # use little endian
        nd_bytes = nd.newbyteorder('<').tobytes("C")
        arr.extend(set_int(len(nd_bytes)))
        arr.extend(nd_bytes)  # make it big endian
    return arr


def _shape_encode(shape: tuple, arr: bytearray):
    arr.extend(set_int(len(shape)))
    layout = ""
    for ele in shape:
        arr.extend(set_long(ele))
        layout += "?"
    arr.extend(set_int(len(layout)))
    for ele in layout:
        arr.extend(set_char(ele))


def _shape_decode(encoded: bytearray, idx: int) -> tuple:
    length, idx = get_int(encoded, idx)
    shape = []
    for _ in range(length):
        dim, idx = get_long(encoded, idx)
        shape.append(dim)
    layout_len, idx = get_int(encoded, idx)
    for _ in range(layout_len):
        _, idx = get_char(encoded, idx)
    return tuple(shape), idx
