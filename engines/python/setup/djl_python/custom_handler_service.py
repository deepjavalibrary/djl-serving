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
from djl_python.inputs import Input
from djl_python.service_loader import load_model_service, has_function_in_module
from djl_python.async_utils import create_non_stream_output

logger = logging.getLogger(__name__)


class CustomHandlerError(Exception):
    """Exception raised when custom handler code fails"""

    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception
        self.__cause__ = original_exception


class CustomHandlerService:

    def __init__(self, properties: dict):
        self.custom_handler = None
        self.initialized = False
        self._initialize(properties)

    def _initialize(self, properties: dict):
        model_dir = properties.get("model_dir", ".")
        try:
            service = load_model_service(model_dir, "model.py", -1)
            if has_function_in_module(service.module, "handle"):
                self.custom_handler = service
                logger.info("Loaded custom handler from model.py")
                self.initialized = True
        except Exception as e:
            logger.debug(f"No custom handler found in model.py: {e}")

    async def handle(self, inputs: Input):
        if self.custom_handler:
            try:
                return await self.custom_handler.invoke_handler_async(
                    "handle", inputs)
            except Exception as e:
                logger.exception("Custom handler failed")
                output = create_non_stream_output(
                    "", error=f"Custom handler failed: {str(e)}", code=424)
                return output
        return None