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
from typing import Optional, AsyncGenerator, Any, Dict

from djl_python.custom_formatter_handling import CustomFormatterHandler
from djl_python.adapter_manager_mixin import AdapterManagerMixin
from djl_python.inputs import Input
from djl_python.outputs import Output

logger = logging.getLogger(__name__)


class AdapterFormatterMixin(CustomFormatterHandler, AdapterManagerMixin):
    """
    Combined mixin that provides both adapter management and adapter-aware formatter application.
    
    This class inherits from both CustomFormatterHandler and AdapterManagerMixin, providing
    a single base class for services that need adapter support with custom formatters.
    
    Responsibilities:
    - Base model formatter functionality (from CustomFormatterHandler)
    - Adapter weight registration/management (from AdapterManagerMixin)
    - Custom code management (loading/unloading adapter-specific formatters)
    - Adapter-aware formatter application (methods in this class)
    """

    def __init__(self):
        CustomFormatterHandler.__init__(self)
        AdapterManagerMixin.__init__(self)
        self.adapter_code_registry: Dict[str, CustomFormatterHandler] = {}

    def get_adapter_formatter_handler(
            self, adapter_name: str) -> Optional[CustomFormatterHandler]:
        """
        Retrieves the formatter handler for a specific adapter.
        
        Args:
            adapter_name: Unique identifier for the adapter
            
        Returns:
            CustomFormatterHandler if adapter has custom code, None otherwise
        """
        return self.adapter_code_registry.get(adapter_name)

    def apply_input_formatter(self,
                              decoded_payload: Any,
                              adapter_name: Optional[str] = None,
                              **kwargs) -> Any:
        """
        Override to apply input formatter, using adapter-specific formatter if available.
        
        Args:
            decoded_payload: The decoded request payload
            adapter_name: Optional adapter name to use for custom formatter
            **kwargs: Additional arguments to pass to the formatter
            
        Returns:
            Formatted input
        """
        # Check if adapter has custom formatter
        if adapter_name:
            adapter_formatter = self.get_adapter_formatter_handler(
                adapter_name)
            if adapter_formatter and adapter_formatter.input_formatter:
                logger.debug(
                    f"Using adapter-specific input formatter for adapter '{adapter_name}'"
                )
                return adapter_formatter.apply_input_formatter(
                    decoded_payload, **kwargs)

        # Use base model formatter
        logger.debug("Using base model input formatter")
        return super().apply_input_formatter(decoded_payload, **kwargs)

    def apply_output_formatter(self,
                               output: Any,
                               adapter_name: Optional[str] = None,
                               **kwargs) -> Any:
        """
        Override to apply output formatter, using adapter-specific formatter if available.
        
        Args:
            output: The model output
            adapter_name: Optional adapter name to use for custom formatter
            **kwargs: Additional arguments to pass to the formatter
            
        Returns:
            Formatted output
        """
        # Check if adapter has custom formatter
        if adapter_name:
            adapter_formatter = self.get_adapter_formatter_handler(
                adapter_name)
            if adapter_formatter and adapter_formatter.output_formatter:
                logger.debug(
                    f"Using adapter-specific output formatter for adapter '{adapter_name}'"
                )
                return adapter_formatter.apply_output_formatter(
                    output, **kwargs)

        # Use base model formatter
        logger.debug("Using base model output formatter")
        return super().apply_output_formatter(output, **kwargs)

    async def apply_output_formatter_streaming_raw(
            self,
            response: AsyncGenerator,
            adapter_name: Optional[str] = None,
            **kwargs) -> AsyncGenerator:
        """
        Override to apply streaming output formatter, using adapter-specific formatter if available.
        
        Args:
            response: The async generator producing model outputs
            adapter_name: Optional adapter name to use for custom formatter
            **kwargs: Additional arguments to pass to the formatter
            
        Returns:
            Async generator with formatted outputs
        """
        # Check if adapter has custom formatter
        if adapter_name:
            adapter_formatter = self.get_adapter_formatter_handler(
                adapter_name)
            if adapter_formatter and adapter_formatter.output_formatter:
                logger.debug(
                    f"Using adapter-specific streaming output formatter for adapter '{adapter_name}'"
                )
                async for item in adapter_formatter.apply_output_formatter_streaming_raw(
                        response, **kwargs):
                    yield item
                return

        # Use base model formatter
        logger.debug("Using base model streaming output formatter")
        async for item in super().apply_output_formatter_streaming_raw(
                response, **kwargs):
            yield item

    def load_adapter_custom_code(self, adapter_name: str,
                                 adapter_path: str) -> CustomFormatterHandler:
        """
        Load custom code (model.py) for an adapter.
        
        Args:
            adapter_name: Unique identifier for the adapter
            adapter_path: Path to adapter directory containing model.py
            
        Returns:
            CustomFormatterHandler instance with loaded formatters
            
        Raises:
            FileNotFoundError: If model.py doesn't exist
            ValueError: If custom code loading fails
        """
        model_py_path = os.path.join(adapter_path, "model.py")

        if not os.path.isfile(model_py_path):
            raise FileNotFoundError(
                f"model.py not found in adapter directory: {adapter_path}")

        logger.info(
            f"Loading custom code for adapter '{adapter_name}' from {model_py_path}"
        )

        try:
            # Create a new CustomFormatterHandler and load formatters from model.py
            # Pass adapter_name as namespace for unique module naming
            formatter_handler = CustomFormatterHandler()
            formatter_handler.load_formatters(adapter_path,
                                              namespace=adapter_name)

            # Store in registry
            self.adapter_code_registry[adapter_name] = formatter_handler

            logger.info(
                f"Successfully loaded custom code for adapter '{adapter_name}'"
            )
            return formatter_handler

        except Exception as e:
            logger.exception(
                f"Failed to load custom code for adapter '{adapter_name}'")
            raise ValueError(
                f"Failed to load custom code for adapter {adapter_name}: {str(e)}"
            )

    def unload_adapter_custom_code(self, adapter_name: str) -> bool:
        """
        Unload custom code for an adapter.
        
        Args:
            adapter_name: Unique identifier for the adapter
            
        Returns:
            True if custom code was unloaded, False if no custom code was loaded
        """
        if adapter_name not in self.adapter_code_registry:
            logger.debug(
                f"Adapter '{adapter_name}' not found in code registry")
            return False

        logger.info(f"Unloading custom code for adapter '{adapter_name}'")
        del self.adapter_code_registry[adapter_name]

        return True

    async def register_adapter(self, inputs: Input) -> Output:
        """
        Override register_adapter to handle custom code loading.
        
        This method extends the base AdapterManagerMixin.register_adapter to add
        custom code management before adapter weight loading.
        """
        adapter_name = inputs.get_property("name")
        adapter_alias = inputs.get_property("alias") or adapter_name
        adapter_path = inputs.get_property("src")

        # Check for custom code and load it BEFORE registering adapter weights
        model_py_path = os.path.join(adapter_path, "model.py")
        if os.path.isfile(model_py_path):
            try:
                self.load_adapter_custom_code(adapter_name, adapter_path)
            except Exception as e:
                # Fail fast - don't load adapter weights if custom code fails
                outputs = Output()
                err = {"data": "", "last": True, "code": 424, "error": str(e)}
                outputs.add(Output.binary_encode(err), key="data")
                return outputs

        # Now register adapter weights using parent implementation
        return await super().register_adapter(inputs)

    async def unregister_adapter(self, inputs: Input) -> Output:
        """
        Override unregister_adapter to handle custom code unloading.
        
        This method extends the base AdapterManagerMixin.unregister_adapter to add
        custom code cleanup after adapter weight unloading.
        """
        adapter_name = inputs.get_property("name")

        # First unregister adapter weights using parent implementation
        result = await super().unregister_adapter(inputs)

        # Then unload custom code if present
        self.unload_adapter_custom_code(adapter_name)

        return result
