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
Python server that is started from DJL serving.
Communication message format: binary encoding
"""

import logging
import os
import socket
import sys

from protocol.request_handler import retrieve_request
from util.arg_parser import ArgParser
from util.serializing import construct_enc_response


class PythonServer(object):
    def __init__(self, sock_type, host=None, port=None, sock_name=None):
        """
        Initializes the socket
        :param host:
        :param port:
        """
        self.sock_type = sock_type
        if sock_type == "unix":
            if sock_name is None:
                raise ValueError("Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = sock_name, -1
            try:
                os.remove(sock_name)
            except OSError:
                if os.path.exists(sock_name):
                    raise RuntimeError("socket already in use: {}.".format(sock_name))
        elif sock_type == "tcp":
            self.sock_name = host if host is not None else "127.0.0.1"
            if port is None:
                raise ValueError("Wrong arguments passed. No socket port given.")
            self.port = int(port)
        else:
            raise ValueError("Invalid socket type provided")

        socket_family = socket.AF_INET if sock_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_name = sock_name

    def run_server(self):
        """
        Starts the server and listens
        """
        if self.sock_type == "unix":
            self.sock.bind(self.sock_name)
        else:
            self.sock.bind((self.sock_name, self.port))
        self.sock.listen(128)
        logging.info("[PID] %d", os.getpid())
        logging.info("Python server started.")
        cl_sock, _ = self.sock.accept()
        logging.info("DJL Client is connected.")

        while True:
            byte_data = retrieve_request(cl_sock)
            logging.info("Received request from DJL Client")
            response_data = construct_enc_response(byte_data)
            is_sent = cl_sock.sendall(response_data)
            if not is_sent:
                logging.info("Response is sent to DJL Client")


if __name__ == "__main__":
    try:
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
        args = ArgParser.python_server_args().parse_args()
        sock_type = args.sock_type
        host = args.host
        port = args.port
        sock_name = args.sock_name
        server = PythonServer(sock_type=sock_type, host=host, port=port, sock_name=sock_name)
        server.run_server()
    except socket.timeout:
        logging.error("Python server did not receive connection")
    except Exception:
        logging.error("Python server died", exc_info=True)
    exit(1)
