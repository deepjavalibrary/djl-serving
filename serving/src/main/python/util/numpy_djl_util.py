import struct
from typing import List, Tuple

import numpy as np

from util.codec_utils import get_int, get_long, get_char, get_str, get_byte_as_int, get_bytes, set_int, set_long, \
    set_char, set_str

MAGIC_NUMBER = "NDAR"
VERSION = 2


def djl_to_np_decode(encoded: bytearray) -> List[np.ndarray]:
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


def np_to_djl_encode(ndlist: List[np.ndarray]) -> bytearray:
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


def _shape_encode(shape: Tuple[int], arr: bytearray):
    arr.extend(set_int(len(shape)))
    layout = ""
    for ele in shape:
        arr.extend(set_long(ele))
        layout += "?"
    arr.extend(set_int(len(layout)))
    for ele in layout:
        arr.extend(set_char(ele))


def _shape_decode(encoded: bytearray, idx: int) -> Tuple[Tuple, int]:
    length, idx = get_int(encoded, idx)
    shape = []
    for _ in range(length):
        dim, idx = get_long(encoded, idx)
        shape.append(dim)
    layout_len, idx = get_int(encoded, idx)
    for _ in range(layout_len):
        _, idx = get_char(encoded, idx)
    return tuple(shape), idx
