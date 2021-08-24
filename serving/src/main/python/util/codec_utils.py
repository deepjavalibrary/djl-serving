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

from protocol.input import Input
from util.binary_util import _set_int, _get_str, _get_int, _get_bytes
from util.pair_list import PairList


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


def decode_input(arr: bytearray) -> Input:
    idx = 0
    _input = Input()
    req_id, idx = _get_str(arr, idx)
    _input.set_request_id(req_id)
    prop_size, idx = _get_int(arr, idx)

    for _ in range(prop_size):
        key, idx = _get_str(arr, idx)
        val, idx = _get_str(arr, idx)
        _input.add_property(key, val)

    content_size, idx = _get_int(arr, idx)
    keys = []
    for _ in range(content_size):
        key, idx = _get_str(arr, idx)
        keys.append(key)

    values = []
    for _ in range(content_size):
        val_len, idx = _get_int(arr, idx)
        val, idx = _get_bytes(arr, idx, val_len)
        values.append(val)

    content = PairList(keys=keys, values=values)
    _input.set_content(content)
    return _input
