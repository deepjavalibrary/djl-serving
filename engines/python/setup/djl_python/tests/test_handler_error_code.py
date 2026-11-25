#!/usr/bin/env python
#
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import json
import unittest
from concurrent.futures.process import BrokenProcessPool

from djl_python.custom_formatter_handling import CustomFormatterError
from djl_python.utils import handler_error_code


class VLLMClientError(Exception):
    pass


class VLLMValidationError(VLLMClientError):
    pass


class VLLMServerError(Exception):
    pass


class TestHandlerErrorCode(unittest.TestCase):

    def test_request_errors(self):
        try:
            json.loads("{")
        except ValueError as e:
            decode_error = e
        for e in (ValueError("bad input"), decode_error,
                  CustomFormatterError("formatter failed", KeyError("x")),
                  VLLMValidationError("prompt too long")):
            self.assertEqual(handler_error_code(e), 424, repr(e))

    def test_server_errors(self):
        try:
            raise RuntimeError("compile failed") from BrokenProcessPool(
                "A child process terminated abruptly")
        except RuntimeError as e:
            compile_error = e
        for e in (BrokenProcessPool("A child process terminated abruptly"),
                  compile_error, RuntimeError("engine died"),
                  VLLMServerError("generation failed"), KeyError("x")):
            self.assertEqual(handler_error_code(e), 500, repr(e))


if __name__ == '__main__':
    unittest.main()
