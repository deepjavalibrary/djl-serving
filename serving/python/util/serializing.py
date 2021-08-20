import struct


def _set_int(value: int) -> bytes:
    return struct.pack(">i", value)


def construct_enc_response(arr: bytearray) -> bytearray:
    response = bytearray()
    response.extend(_set_int(len(arr)))
    response.extend(arr)
    return response
