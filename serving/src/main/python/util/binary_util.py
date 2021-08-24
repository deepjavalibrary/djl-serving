import struct


def _set_int(value: int) -> bytes:
    """
    Converts int value into bytes.
    :param value: integer
    :return: converted bytes
    """
    return struct.pack(">i", value)


def _set_long(value: int) -> bytes:
    return struct.pack(">q", value)


def _set_str(value: str) -> bytes:
    return struct.pack(">h", len(value)) + bytes(value, "utf8")


def _set_char(value: str) -> bytes:
    return struct.pack('>h', ord(value))


def _get_byte_as_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    return encoded[idx], idx + 1


def _get_bytes(encoded: bytearray, idx: int, length: int) -> tuple[bytes, int]:
    return encoded[idx:idx + length], idx + length


def _get_char(encoded: bytearray, idx: int) -> tuple[str, int]:
    chr_size = 2
    return chr(struct.unpack(">h", encoded[idx:idx + chr_size])[0]), idx + chr_size


def _get_str(encoded: bytearray, idx: int) -> tuple[str, int]:
    length = struct.unpack(">h", encoded[idx:idx + 2])[0]
    idx += 2
    return encoded[idx:idx + length].decode("utf8"), idx + length


def _get_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    int_size = 4
    return struct.unpack(">i", encoded[idx:idx + int_size])[0], idx + int_size


def _get_long(encoded: bytearray, idx: int) -> tuple[int, int]:
    long_size = 8
    return struct.unpack(">q", encoded[idx:idx + long_size])[0], idx + long_size
