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
Argument parser for Python engine.
"""

import argparse


class ArgParser(object):

    @staticmethod
    def python_engine_args():
        """
        Argument parser for Python engine.
        """
        parser = argparse.ArgumentParser(prog='djl_python_engine',
                                         description='DJL Python engine')
        parser.add_argument('--model-dir',
                            required=True,
                            type=str,
                            dest="model_dir",
                            help='Model directory')
        parser.add_argument('--entry-point',
                            required=False,
                            type=str,
                            dest="entry_point",
                            help='The python entry point file')
        parser.add_argument(
            '--sock-type',
            required=True,
            type=str,
            dest="sock_type",
            choices=["unix", "tcp"],
            help=
            'Socket type the python server worker would use. The options are\n'
            'unix: The python server expects to unix domain-socket\n'
            'tcp: The python server expects a host-name and port-number')
        parser.add_argument(
            '--sock-name',
            required=False,
            dest="sock_name",
            type=str,
            help=
            'If \'sock-type\' is \'unix\', sock-name is expected to be a string. '
            'Eg: --sock-name \"test_sock\"')
        parser.add_argument('--device-id',
                            type=str,
                            dest='device_id',
                            required=False,
                            default="-1",
                            help='The device id for the model')
        parser.add_argument(
            '--port',
            type=str,
            help=
            'If \'sock-type\' is \'tcp\' this is expected to have the host port to bind on'
        )
        parser.add_argument('--tensor-parallel-degree',
                            required=False,
                            dest="tensor_parallel_degree",
                            type=int,
                            help='The tensor parallel degree')
        return parser

    @staticmethod
    def test_model_args():
        parser = argparse.ArgumentParser(prog='djl-test-model',
                                         description='Test DJL Python model')
        parser.add_argument('--model-dir',
                            type=str,
                            dest='model_dir',
                            help='Model directory')
        parser.add_argument('--entry-point',
                            required=False,
                            type=str,
                            dest="entry_point",
                            default="model.py",
                            help='The model entry point file')
        parser.add_argument('--handler',
                            type=str,
                            dest='handler',
                            required=False,
                            default="handle",
                            help='Python function to invoke')
        parser.add_argument('--input',
                            type=str,
                            dest='input',
                            required=False,
                            nargs='+',
                            default='input.txt',
                            help='Input file')
        parser.add_argument('--parameters',
                            type=str,
                            dest='parameters',
                            required=False,
                            nargs='+',
                            help='Model parameters')
        return parser
