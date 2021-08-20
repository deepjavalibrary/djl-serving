# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
# http://aws.amazon.com/apache2.0/
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

"""
Contains util functions for serializing data.
"""
import struct


def _set_int(value: int) -> bytes:
    """
    Converts int value into bytes.
    :param value: integer
    :return: converted bytes
    """
    return struct.pack(">i", value)


def construct_enc_response(arr: bytearray) -> bytearray:
    """
    Constructs the response to be sent. length of the array + data.
    :param arr: bytearray
    :return: response format
    """
    response = bytearray()
    response.extend(_set_int(len(arr)))
    response.extend(arr)
    return response
