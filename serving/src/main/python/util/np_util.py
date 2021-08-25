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

from util.binary_util import get_long, get_char, set_int, set_str, get_str, get_int, get_byte_as_int, get_bytes, \
    set_long, set_char

MAGIC_NUMBER = "NDAR"
VERSION = 2


def djl_to_np_decode(encoded: bytearray) -> list[np.ndarray]:
    """
    Converts djl format to list of numpy ndarray
    :param encoded: bytearray
    :return: list of numpy array
    """
    idx = 0
    num_ele, idx = get_int(encoded, idx)
    result = []
    for _ in range(num_ele):
        magic, idx = get_str(encoded, idx)
        if magic != MAGIC_NUMBER:
            raise AssertionError("magic number is not NDAR, actual " + magic)
        version, idx = get_int(encoded, idx)
        if version != VERSION:
            raise AssertionError("require version 2, actual " + str(version))
        flag, idx = get_byte_as_int(encoded, idx)
        if flag == 1:
            _, idx = get_str(encoded, idx)
        _, idx = get_str(encoded, idx)  # ignore sparse format
        datatype, idx = get_str(encoded, idx)
        shape, idx = _shape_decode(encoded, idx)
        data_length, idx = get_int(encoded, idx)
        data, idx = get_bytes(encoded, idx, data_length)
        result.append(np.ndarray(shape, np.dtype([('big', datatype.lower())]), data))
    return result


def np_to_djl_encode(ndlist: list[np.ndarray]) -> bytearray:
    """
    Converts list of numpy ndarray into list to djl ndlist

    :param ndlist: list of numpy ndarray
    :return: djl ndlist as bytearray
    """
    arr = bytearray()
    arr.extend(set_int(len(ndlist)))
    for nd in ndlist:
        arr.extend(set_str(MAGIC_NUMBER))
        arr.extend(set_int(VERSION))
        arr.append(0)  # no name
        arr.extend(set_str("default"))
        arr.extend(set_str(str(nd.dtype[0]).upper()))
        _shape_encode(nd.shape, arr)
        nd_bytes = nd.newbyteorder('>').tobytes("C")
        arr.extend(set_int(len(nd_bytes)))
        arr.extend(nd_bytes)  # make it big endian
    return arr


def _shape_encode(shape: tuple[int], arr: bytearray):
    arr.extend(set_int(len(shape)))
    layout = ""
    for ele in shape:
        arr.extend(set_long(ele))
        layout += "?"
    arr.extend(set_int(len(layout)))
    for ele in layout:
        arr.extend(set_char(ele))


def _shape_decode(encoded: bytearray, idx: int) -> tuple[tuple, int]:
    length, idx = get_int(encoded, idx)
    shape = []
    for _ in range(length):
        dim, idx = get_long(encoded, idx)
        shape.append(dim)
    layout_len, idx = get_int(encoded, idx)
    for _ in range(layout_len):
        _, idx = get_char(encoded, idx)
    return tuple(shape), idx
