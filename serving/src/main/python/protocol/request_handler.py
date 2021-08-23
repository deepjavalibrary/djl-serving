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
Request handler for the python server.
"""

import logging
import struct

from protocol.request import Request

short_size = 2
int_size = 4


def _retrieve_buffer(conn, length):
    """
    Retrieves buffer in the specified length.
    :param conn: socket connection
    :param length: length of the data to be read
    :return: retrieved byter array
    """
    data = bytearray()

    while length > 0:
        pkt = conn.recv(length)
        if len(pkt) == 0:
            logging.info("Frontend disconnected.")
            raise ValueError("Frontend disconnected")

        data += pkt
        length -= len(pkt)

    return data


def _retrieve_int(conn):
    """
    Retrieves int value.
    :param conn: socket connection
    :return: retrieved integer value
    """
    data = _retrieve_buffer(conn, int_size)
    return struct.unpack("!i", data)[0]


def _retrieve_str(conn):
    length_buf = _retrieve_buffer(conn, short_size)
    length = struct.unpack(">h", length_buf)[0]
    data = _retrieve_buffer(conn, length)
    return data.decode("utf8")


def retrieve_request(conn):
    """
    Retrieves the request data.
    :param conn: socket connection

    :return: request object
    """
    request = Request()

    process_type_code = _retrieve_int(conn)
    request.set_request_type(process_type_code)

    python_file = _retrieve_str(conn)
    request.set_python_file(python_file)

    function_name = _retrieve_str(conn)
    request.set_function_name(function_name)

    function_param_len = _retrieve_int(conn)
    function_param = _retrieve_buffer(conn, function_param_len)
    request.set_function_param(function_param)

    return request
