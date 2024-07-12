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
from djl_python.service_loader import load_model_service, has_function_in_module, get_annotated_function
from djl_python.sm_log_filter import SMLogFilter

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

        self.model_dir = args.model_dir
        self.sock_type = args.sock_type
        self.sock_name = args.sock_name
        self.port = args.port
        self.service = service
        self.device_id = args.device_id
        self.tensor_parallel_degree = args.tensor_parallel_degree
        self.pipeline_parallel_degree = args.pipeline_parallel_degree
        self.cluster_size = args.cluster_size
        self.entry_point = args.entry_point
        self.recommended_entry_point = args.recommended_entry_point
        self.input_formatter = get_annotated_function(args.model_dir,
                                                      "is_input_formatter")

        if self.sock_type == "unix":
            if self.sock_name is None:
                raise ValueError("Missing sock-name argument.")
            self.sock_name = f"{args.sock_name}.{rank}" if rank else args.sock_name

            self.clean_up()
        elif self.sock_type == "tcp":
            if self.sock_name is None:
                self.sock_name = "0.0.0.0"
            if self.port is None:
                raise ValueError("Missing port argument.")
            self.port = int(self.port) + int(rank) if rank else self.port
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
            logging.info(
                f"Socket bind on address: {self.sock_name}:{self.port}")
            self.sock.bind((self.sock_name, int(self.port)))

        self.sock.listen(128)
        logging.info("Python engine started.")

        (cl_socket, _) = self.sock.accept()
        # workaround error(35, 'Resource temporarily unavailable') on OSX
        cl_socket.setblocking(True)

        is_entry_point_verified = False
        while True:
            inputs = Input()
            inputs.read(cl_socket)
            prop = inputs.get_properties()
            if self.tensor_parallel_degree:
                prop["tensor_parallel_degree"] = self.tensor_parallel_degree
            if self.pipeline_parallel_degree:
                prop["pipeline_parallel_degree"] = self.pipeline_parallel_degree
            if self.cluster_size:
                prop["cluster_size"] = self.cluster_size
            prop["device_id"] = self.device_id
            if "output_formatter" in prop and hasattr(
                    self.service, prop["output_formatter"]):
                prop["output_formatter"] = getattr(self.service,
                                                   prop["output_formatter"])
            if self.input_formatter:
                prop["input_formatter"] = self.input_formatter
            function_name = inputs.get_function_name()
            if not is_entry_point_verified:
                if self.recommended_entry_point:
                    if not has_function_in_module(self.service.module,
                                                  function_name):
                        self.service = load_model_service(
                            self.model_dir, self.recommended_entry_point,
                            self.device_id)
                        logging.info(
                            f"{self.entry_point} file has no handler function {function_name}."
                            f"Hence choosing the LMI recommended entry point {self.recommended_entry_point}"
                        )
                is_entry_point_verified = True

            try:
                outputs = self.service.invoke_handler(function_name, inputs)
                if outputs is None:
                    outputs = Output(code=204, message="No content")
                elif not isinstance(outputs, Output):
                    outputs = Output().error(
                        f"Invalid output type: {type(outputs)}")
            except Exception as e:
                logging.exception("Failed invoke service.invoke_handler()")
                if type(e).__name__ == "OutOfMemoryError" or type(
                        e).__name__ == "MemoryError":
                    outputs = Output(code=507, message=str(e))
                else:
                    outputs = Output().error(str(e))

            outputs.send(cl_socket)
            logging.debug("Outputs is sent to DJL engine.")
            try:
                outputs.execute_finalize()
            except Exception as e:
                logging.exception(f"Failed on finalize function: {e}")


def main():
    sock_type = None
    sock_name = None
    pid = os.getpid()

    # noinspection PyBroadException
    try:
        logging.basicConfig(stream=sys.stdout,
                            format="%(message)s",
                            level=logging.INFO)
        configure_sm_logging()
        logging.info(
            f"{pid} - djl_python_engine started with args: {sys.argv[1:]}")
        args = ArgParser.python_engine_args().parse_args()
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        sock_type = args.sock_type
        sock_name = args.sock_name if rank is None else f"{args.sock_name}.{rank}"

        entry_point = args.entry_point if args.entry_point else args.recommended_entry_point
        model_service = load_model_service(args.model_dir, entry_point,
                                           args.device_id)

        engine = PythonEngine(args, model_service)

        engine.run_server()
    except socket.timeout:
        logging.error(f"Listener timed out in: {SOCKET_ACCEPT_TIMEOUT} s.")
    except Exception:  # pylint: disable=broad-except
        logging.exception("Python engine process died")
    finally:
        logging.info(f"{pid} - Python process finished")
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        if sock_type == 'unix':
            if os.path.exists(sock_name):
                os.remove(sock_name)
            pid_file = f"{sock_name}.pid"
            if os.path.exists(pid_file):
                os.remove(pid_file)


def configure_sm_logging():
    if 'SM_TELEMETRY_LOG_REV_2022_12' in os.environ:
        # https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/logging-and-monitoring.html
        root_logger = logging.getLogger()
        sm_log_handler = logging.FileHandler(
            filename=os.getenv('SM_TELEMETRY_LOG_REV_2022_12'))
        sm_log_handler.addFilter(SMLogFilter())
        root_logger.addHandler(sm_log_handler)


if __name__ == "__main__":
    main()
    exit(1)
