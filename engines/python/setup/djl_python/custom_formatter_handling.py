#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging

from djl_python.service_loader import get_annotated_function

logger = logging.getLogger(__name__)


class CustomFormatterError(Exception):
    """Exception raised when custom formatter code fails"""

    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception
        self.__cause__ = original_exception


class CustomFormatterHandler:

    def __init__(self):
        self.output_formatter = None
        self.input_formatter = None

    def load_formatters(self, model_dir: str):
        """Load custom formatters from model.py"""
        try:
            self.input_formatter = get_annotated_function(
                model_dir, "is_input_formatter")
            self.output_formatter = get_annotated_function(
                model_dir, "is_output_formatter")
            logger.info(
                f"Loaded formatters - input: {self.input_formatter}, output: {self.output_formatter}"
            )
        except Exception as e:
            raise CustomFormatterError(
                f"Failed to load custom formatters from {model_dir}", e)

    def apply_input_formatter(self, decoded_payload, **kwargs):
        """Apply input formatter if available"""
        if self.input_formatter:
            try:
                return self.input_formatter(decoded_payload, **kwargs)
            except Exception as e:
                logger.exception("Custom input formatter failed")
                raise CustomFormatterError(
                    "Custom input formatter execution failed", e)
        return decoded_payload

    def apply_output_formatter(self, output):
        """Apply output formatter if available"""
        if self.output_formatter:
            try:
                return self.output_formatter(output)
            except Exception as e:
                logger.exception("Custom output formatter failed")
                raise CustomFormatterError(
                    "Custom output formatter execution failed", e)
        return output

    async def apply_output_formatter_streaming_raw(self, stream_generator):
        """Apply output formatter to raw streaming responses"""
        try:
            async for response in stream_generator:
                yield self.apply_output_formatter(response)
        except CustomFormatterError:
            raise
        except Exception as e:
            logger.exception("Streaming formatter failed")
            raise CustomFormatterError(
                "Custom streaming formatter execution failed", e)
