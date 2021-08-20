# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
# http://aws.amazon.com/apache2.0/
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
import logging
import os
import socket
import sys

from protocol.request_handler import retrieve_request
from util.arg_parser import ArgParser
from util.serializing import construct_enc_response


class SocketServer(object):
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_name = host
        self.port = int(port)

    def run_server(self):
        self.sock.bind((self.sock_name, self.port))
        self.sock.listen(128)
        logging.info("[PID] %d", os.getpid())
        logging.info("Python server started.")
        cl_sock, _ = self.sock.accept()
        logging.info("DJL Client is connected")

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
        host = args.host
        port = args.port
        server = SocketServer(host, port)
        server.run_server()
    except socket.timeout:
        logging.error("Python server did not receive connection")
    except Exception:
        logging.error("Python server died", exc_info=True)
    exit(1)
