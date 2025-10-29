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
import os
from dataclasses import dataclass
from typing import Optional, Callable

from djl_python.service_loader import get_annotated_function
from djl_python.utils import get_sagemaker_function

logger = logging.getLogger(__name__)


class CustomFormatterError(Exception):
    """Exception raised when custom formatter code fails"""

    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception
        self.__cause__ = original_exception


@dataclass
class CustomFormatters:
    """Container for input/output formatting functions"""
    input_formatter: Optional[Callable] = None
    output_formatter: Optional[Callable] = None


@dataclass
class CustomHandlers:
    """Container for prediction/initialization handler functions"""
    prediction_handler: Optional[Callable] = None
    init_handler: Optional[Callable] = None


@dataclass
class CustomCode:
    """Container for all custom formatters and handlers"""
    formatters: CustomFormatters
    handlers: CustomHandlers
    is_sagemaker_script: bool = False

    def __init__(self):
        self.formatters = CustomFormatters()
        self.handlers = CustomHandlers()
        self.is_sagemaker_script = False


class CustomFormatterHandler:

    def __init__(self):
        self.custom_code = CustomCode()

    def load_formatters(self, model_dir: str) -> CustomCode:
        """Load custom formatters/handlers from model.py with SageMaker detection"""
        try:
            self.custom_code.formatters.input_formatter = get_annotated_function(
                model_dir, "is_input_formatter")
            self.custom_code.formatters.output_formatter = get_annotated_function(
                model_dir, "is_output_formatter")
            self.custom_code.handlers.prediction_handler = get_annotated_function(
                model_dir, "is_prediction_handler")
            self.custom_code.handlers.init_handler = get_annotated_function(
                model_dir, "is_init_handler")

            # Detect SageMaker script pattern for backward compatibility
            self._detect_sagemaker_functions(model_dir)

            logger.info(
                f"Loaded formatters - input: {bool(self.custom_code.formatters.input_formatter)}, "
                f"output: {bool(self.custom_code.formatters.output_formatter)}"
            )
            logger.info(
                f"Loaded handlers - prediction: {bool(self.custom_code.handlers.prediction_handler)}, "
                f"init: {bool(self.custom_code.handlers.init_handler)}, "
                f"sagemaker: {self.custom_code.is_sagemaker_script}")
            return self.custom_code
        except Exception as e:
            raise CustomFormatterError(
                f"Failed to load custom code from {model_dir}", e)

    def _detect_sagemaker_functions(self, model_dir: str):
        """Detect and load SageMaker-style functions for backward compatibility"""
        # If no decorator-based code found, check for SageMaker functions
        if not any([
                self.custom_code.formatters.input_formatter,
                self.custom_code.formatters.output_formatter,
                self.custom_code.handlers.prediction_handler,
                self.custom_code.handlers.init_handler
        ]):
            sagemaker_model_fn = get_sagemaker_function(model_dir, 'model_fn')
            sagemaker_input_fn = get_sagemaker_function(model_dir, 'input_fn')
            sagemaker_predict_fn = get_sagemaker_function(
                model_dir, 'predict_fn')
            sagemaker_output_fn = get_sagemaker_function(
                model_dir, 'output_fn')

            if any([
                    sagemaker_model_fn, sagemaker_input_fn,
                    sagemaker_predict_fn, sagemaker_output_fn
            ]):
                self.custom_code.is_sagemaker_script = True
                if sagemaker_model_fn:
                    self.custom_code.handlers.init_handler = sagemaker_model_fn
                if sagemaker_input_fn:
                    self.custom_code.formatters.input_formatter = sagemaker_input_fn
                if sagemaker_predict_fn:
                    self.custom_code.handlers.prediction_handler = sagemaker_predict_fn
                if sagemaker_output_fn:
                    self.custom_code.formatters.output_formatter = sagemaker_output_fn
                logger.info("Loaded SageMaker-style functions")

    def apply_input_formatter(self, decoded_payload, **kwargs):
        """Apply input formatter if available"""
        if self.custom_code.formatters.input_formatter:
            try:
                return self.custom_code.formatters.input_formatter(
                    decoded_payload, **kwargs)
            except Exception as e:
                logger.exception("Custom input formatter failed")
                raise CustomFormatterError(
                    "Custom input formatter execution failed", e)
        return decoded_payload

    def apply_output_formatter(self, output):
        """Apply output formatter if available"""
        if self.custom_code.formatters.output_formatter:
            try:
                return self.custom_code.formatters.output_formatter(output)
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


def load_custom_code(model_dir: str) -> CustomCode:
    """Load custom code, checking DJL decorators first, then SageMaker functions"""
    handler = CustomFormatterHandler()
    return handler.load_formatters(model_dir)
