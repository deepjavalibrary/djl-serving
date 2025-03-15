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
import socket
import sys

from djl_python.arg_parser import ArgParser
from djl_python.service_loader import load_model_service
from djl_python.sm_log_filter import SMLogFilter
from djl_python.python_sync_engine import PythonSyncEngine
from djl_python.python_async_engine import PythonAsyncEngine

SOCKET_ACCEPT_TIMEOUT = 30.0


def main():
    sock_type = None
    sock_name = None
    pid = os.getpid()

    # noinspection PyBroadException
    try:
        args = ArgParser.python_engine_args().parse_args()
        logging.basicConfig(stream=sys.stdout,
                            format="%(levelname)s::%(message)s",
                            level=args.log_level.upper())
        configure_sm_logging()
        logging.info(
            f"{pid} - djl_python_engine started with args: {sys.argv[1:]}")
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        sock_type = args.sock_type
        sock_name = args.sock_name if rank is None else f"{args.sock_name}.{rank}"

        entry_point = args.entry_point if args.entry_point else args.recommended_entry_point
        model_service = load_model_service(args.model_dir, entry_point,
                                           args.device_id)

        if args.async_mode:
            engine = PythonAsyncEngine(args, model_service)
        else:
            engine = PythonSyncEngine(args, model_service)

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
