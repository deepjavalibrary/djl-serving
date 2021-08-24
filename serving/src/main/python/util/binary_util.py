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
import struct


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


def get_byte_as_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    """
    Returns a byte value as int and next to be read index
    :param encoded: bytearray
    :param idx: index to lookup
    :return: tuple of byte read and next index
    """
    return encoded[idx], idx + 1


def get_bytes(encoded: bytearray, idx: int, length: int) -> tuple[bytes, int]:
    """
    Returns the bytes with given length, and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :param length: length to be read
    :return: typle of bytes read and next index
    """
    return encoded[idx:idx + length], idx + length


def get_char(encoded: bytearray, idx: int) -> tuple[str, int]:
    """
    Returns a char and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of char and next index
    """
    chr_size = 2
    return chr(struct.unpack(">h", encoded[idx:idx + chr_size])[0]), idx + chr_size


def get_str(encoded: bytearray, idx: int) -> tuple[str, int]:
    """
    Returns a string and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of string and next index
    """
    length = struct.unpack(">h", encoded[idx:idx + 2])[0]
    idx += 2
    return encoded[idx:idx + length].decode("utf8"), idx + length


def get_int(encoded: bytearray, idx: int) -> tuple[int, int]:
    """
    Returns an integer and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: tuple of int and next index
    """
    int_size = 4
    return struct.unpack(">i", encoded[idx:idx + int_size])[0], idx + int_size


def get_long(encoded: bytearray, idx: int) -> tuple[int, int]:
    """
    Returns the long value and next to be read index
    :param encoded: bytearray
    :param idx: index to start read from
    :return: typle of long and next index
    """
    long_size = 8
    return struct.unpack(">q", encoded[idx:idx + long_size])[0], idx + long_size
