import numpy as np

from util.binary_util import _get_long, _get_char, _set_int, _set_str, _get_str, _get_int, _get_byte_as_int, _get_bytes, \
    _set_long, _set_char

MAGIC_NUMBER = "NDAR"
VERSION = 2


def djl_to_np_decode(encoded: bytearray) -> list[np.ndarray]:
    idx = 0
    num_ele, idx = _get_int(encoded, idx)
    result = []
    for _ in range(num_ele):
        magic, idx = _get_str(encoded, idx)
        if magic != MAGIC_NUMBER:
            raise AssertionError("magic number is not NDAR, actual " + magic)
        version, idx = _get_int(encoded, idx)
        if version != VERSION:
            raise AssertionError("require version 2, actual " + str(version))
        flag, idx = _get_byte_as_int(encoded, idx)
        if flag == 1:
            _, idx = _get_str(encoded, idx)
        _, idx = _get_str(encoded, idx)  # ignore sparse format
        datatype, idx = _get_str(encoded, idx)
        shape, idx = _shape_decode(encoded, idx)
        data_length, idx = _get_int(encoded, idx)
        data, idx = _get_bytes(encoded, idx, data_length)
        result.append(np.ndarray(shape, np.dtype([('big', datatype.lower())]), data))
    return result


def np_to_djl_encode(ndlist: list[np.ndarray]) -> bytearray:
    arr = bytearray()
    arr.extend(_set_int(len(ndlist)))
    for nd in ndlist:
        arr.extend(_set_str(MAGIC_NUMBER))
        arr.extend(_set_int(VERSION))
        arr.append(0)  # no name
        arr.extend(_set_str("default"))
        arr.extend(_set_str(str(nd.dtype[0]).upper()))
        _shape_encode(nd.shape, arr)
        nd_bytes = nd.newbyteorder('>').tobytes("C")
        arr.extend(_set_int(len(nd_bytes)))
        arr.extend(nd_bytes)  # make it big endian
    return arr


def _shape_encode(shape: tuple[int], arr: bytearray):
    arr.extend(_set_int(len(shape)))
    layout = ""
    for ele in shape:
        arr.extend(_set_long(ele))
        layout += "?"
    arr.extend(_set_int(len(layout)))
    for ele in layout:
        arr.extend(_set_char(ele))


def _shape_decode(encoded: bytearray, idx: int) -> tuple[tuple, int]:
    length, idx = _get_int(encoded, idx)
    shape = []
    for _ in range(length):
        dim, idx = _get_long(encoded, idx)
        shape.append(dim)
    layout_len, idx = _get_int(encoded, idx)
    for _ in range(layout_len):
        _, idx = _get_char(encoded, idx)
    return tuple(shape), idx
