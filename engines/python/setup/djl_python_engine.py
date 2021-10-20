#!/usr/bin/env python
#
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
DJL Python engine.
Communication message format: binary encoding
"""

import importlib
import logging
import os
import socket
import sys

from djl_python.arg_parser import ArgParser
from djl_python.inputs import Input
from djl_python.outputs import Output

SOCKET_ACCEPT_TIMEOUT = 30.0


class PythonEngine(object):
    """
    Backend engine to run python code
    """
    def __init__(self, socket_type, socket_name, port, service):
        self.socket_type = socket_type
        self.sock_name = socket_name
        self.port = port
        self.service = service
        if socket_type == "unix":
            if socket_name is None:
                raise ValueError(
                    "Wrong arguments passed. No socket name given.")
            try:
                os.remove(socket_name)
            except OSError:
                if os.path.exists(socket_name):
                    raise RuntimeError(
                        "socket already in use: {}.".format(socket_name))

        elif socket_type == "tcp":
            self.sock_name = "127.0.0.1"
            if port is None:
                raise ValueError(
                    "Wrong arguments passed. No socket port given.")
        else:
            raise ValueError("Invalid socket type provided")

        socket_family = socket.AF_INET if socket_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)

    def run_server(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """
        if self.socket_type == "unix":
            self.sock.bind(self.sock_name)
        else:
            self.sock.bind((self.sock_name, int(self.port)))

        self.sock.listen(128)
        logging.info("Python engine started.")

        (cl_socket, _) = self.sock.accept()
        # workaround error(35, 'Resource temporarily unavailable') on OSX
        cl_socket.setblocking(True)

        while True:
            inputs = Input()
            inputs.read(cl_socket)
            function_name = inputs.get_function_name()
            try:
                outputs = getattr(self.service, function_name)(inputs)
                if outputs is None:
                    outputs = Output(code=204, message="No content")
            except Exception as e:
                logging.error(e, exc_info=True)
                _, ex_value, _ = sys.exc_info()
                outputs = Output(code=500, message=str(ex_value))

            if not cl_socket.sendall(outputs.encode()):
                logging.debug("Outputs is sent to DJL engine.")


if __name__ == "__main__":
    sock_type = None
    sock_name = None

    # noinspection PyBroadException
    try:
        logging.basicConfig(stream=sys.stdout,
                            format="%(message)s",
                            level=logging.INFO)
        logging.info("djl_python_engine started with args: %s",
                     " ".join(sys.argv[1:]))
        args = ArgParser.python_engine_args().parse_args()
        model_dir = args.model_dir
        entry_point = args.entry_point
        entry_point_file = os.path.join(model_dir, entry_point)
        if not os.path.exists(entry_point_file):
            raise ValueError(
                "entry-point file not found {}.".format(entry_point_file))

        entry_point = entry_point[:-3]
        module = importlib.import_module(entry_point)
        if module is None:
            raise ValueError("Unable to load entry_point {}/{}.py".format(
                model_dir, entry_point))

        sock_name = args.sock_name
        sock_type = args.sock_type

        engine = PythonEngine(sock_type, sock_name, args.port, module)

        engine.run_server()
    except socket.timeout:
        logging.error("Python engine did not receive connection in: %d s.",
                      SOCKET_ACCEPT_TIMEOUT)
    except Exception:  # pylint: disable=broad-except
        logging.error("Python engine process died", exc_info=True)
    finally:
        if sock_type == 'unix' and os.path.exists(sock_name):
            os.remove(sock_name)

    exit(1)
