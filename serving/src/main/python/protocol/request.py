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
This class contain the format of the request received.
"""


class Request(object):
    def __init__(self):
        """
        request_type : represents whether it is a preprocess request or postprocess request.
        python_file : path of the python data-process program to be executed.
        function_name : executes this function in the given python program.
        self.function_param : params to the function in encoded binary format.
        """
        self._request_type = None
        self._python_file = None
        self._function_name = None
        self._function_param_bytes = None

    def set_request_type(self, request_type):
        self._request_type = request_type

    def get_request_type(self):
        return self._request_type

    def set_python_file(self, python_file):
        self._python_file = python_file

    def get_python_file(self):
        return self._python_file

    def set_function_name(self, function_name):
        self._function_name = function_name

    def get_function_name(self):
        return self._function_name

    def set_function_param(self, function_param):
        self._function_param_bytes = function_param

    def get_function_param(self):
        return self._function_param_bytes
