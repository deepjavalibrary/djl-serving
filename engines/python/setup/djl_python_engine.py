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

import logging
import os
import signal
import socket
import sys

from djl_python.arg_parser import ArgParser
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.service_loader import load_model_service

SOCKET_ACCEPT_TIMEOUT = 30.0


class PythonEngine(object):
    """
    Backend engine to run python code
    """

    def __init__(self, args, service):
        # Support MPI environment args
        if os.getenv('OMPI_COMM_WORLD_SIZE'):
            os.environ["WORLD_SIZE"] = os.getenv('OMPI_COMM_WORLD_SIZE')
        if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
            os.environ["LOCAL_RANK"] = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        if rank:
            os.environ["RANK"] = rank

        self.sock_type = args.sock_type
        self.sock_name = f"{args.sock_name}.{rank}" if rank else args.sock_name
        self.port = args.port
        self.service = service
        self.device_id = args.device_id
        self.tensor_parallel_degree = args.tensor_parallel_degree

        if self.sock_type == "unix":
            if self.sock_name is None:
                raise ValueError("Missing sock-name argument.")

            self.clean_up()
        elif self.sock_type == "tcp":
            self.sock_name = "127.0.0.1"
            if self.port is None:
                raise ValueError("Missing port argument.")
        else:
            raise ValueError(f"Invalid socket-type: {self.sock_type}.")

        socket_family = socket.AF_INET if self.sock_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)

    def clean_up(self):
        pid_file = f"{self.sock_name}.pid"
        if os.path.exists(pid_file):
            with open(pid_file, "r") as f:
                pid = f.readline()
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logging.warning(
                            f"{self.sock_name} - kill dangling process: {pid}")
                    except ProcessLookupError:
                        pass

        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        if os.path.exists(self.sock_name):
            os.remove(self.sock_name)

    def run_server(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """
        if self.sock_type == "unix":
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
            prop = inputs.get_properties()
            if self.tensor_parallel_degree:
                prop["tensor_parallel_degree"] = self.tensor_parallel_degree
            prop["device_id"] = self.device_id
            function_name = inputs.get_function_name()
            try:
                outputs = self.service.invoke_handler(function_name, inputs)
                if outputs is None:
                    outputs = Output(code=204, message="No content")
            except Exception as e:
                logging.exception("Failed invoke service.invoke_handler()")
                outputs = Output().error(str(e))

            outputs.send(cl_socket)
            logging.debug("Outputs is sent to DJL engine.")


def main():
    sock_type = None
    sock_name = None
    pid = os.getpid()

    # noinspection PyBroadException
    try:
        logging.basicConfig(stream=sys.stdout,
                            format="%(message)s",
                            level=logging.INFO)
        logging.info(
            f"{pid} - djl_python_engine started with args: {sys.argv[1:]}")
        args = ArgParser.python_engine_args().parse_args()
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        sock_type = args.sock_type
        sock_name = args.sock_name if rank is None else f"{args.sock_name}.{rank}"

        model_service = load_model_service(args.model_dir, args.entry_point,
                                           args.device_id)

        engine = PythonEngine(args, model_service)

        engine.run_server()
    except socket.timeout:
        logging.error(f"Listener timed out in: {SOCKET_ACCEPT_TIMEOUT} s.")
    except Exception:  # pylint: disable=broad-except
        logging.exception("Python engine process died")
    finally:
        logging.info(f"{pid} - Python process finished")
        if sock_type == 'unix':
            if os.path.exists(sock_name):
                os.remove(sock_name)
            pid_file = f"{sock_name}.pid"
            if os.path.exists(pid_file):
                os.remove(pid_file)


if __name__ == "__main__":
    main()
    exit(1)
