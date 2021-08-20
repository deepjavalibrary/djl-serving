# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
# http://aws.amazon.com/apache2.0/
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

"""
Argument parser for python server.
"""

import argparse


class ArgParser(object):
    @staticmethod
    def python_server_args():
        """
        Argument parser for python server.
        """
        parser = argparse.ArgumentParser(prog='python-server', description='Python server')
        # TODO: add sock_type as argument
        parser.add_argument('--host',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have a host IP address')
        parser.add_argument('--port',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have the host port to bind on')
        return parser
