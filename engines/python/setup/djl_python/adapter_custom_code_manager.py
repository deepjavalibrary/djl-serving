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
from typing import Dict, Optional

from djl_python.custom_formatter_handling import CustomFormatterHandler

logger = logging.getLogger(__name__)


class AdapterCustomCodeManager:
    """
    Manages adapter-specific custom formatters.
    Keeps it simple - just load, get, and unload.
    """

    def __init__(self):
        """Initialize with simple dict for code registry."""
        self.code_registry: Dict[str, CustomFormatterHandler] = {}
        logger.info("AdapterCustomCodeManager initialized")

    def load_adapter_code(self, adapter_name: str, adapter_dir: str) -> CustomFormatterHandler:
        """
        Loads custom code (model.py) from adapter directory.
        
        Args:
            adapter_name: Unique identifier for the adapter
            adapter_dir: Path to adapter directory containing model.py
            
        Returns:
            CustomFormatterHandler instance with loaded formatters
            
        Raises:
            FileNotFoundError: If model.py doesn't exist
            Exception: If loading fails (import errors, syntax errors, etc.)
        """
        model_py_path = os.path.join(adapter_dir, "model.py")
        
        if not os.path.isfile(model_py_path):
            raise FileNotFoundError(f"model.py not found in adapter directory: {adapter_dir}")
        
        logger.info(f"Loading custom code for adapter '{adapter_name}' from {model_py_path}")
        
        try:
            # Create a new CustomFormatterHandler and load formatters from model.py
            # Pass adapter_name as namespace for unique module naming
            formatter_handler = CustomFormatterHandler()
            formatter_handler.load_formatters(adapter_dir, namespace=adapter_name)
            
            # Store in registry
            self.code_registry[adapter_name] = formatter_handler
            
            logger.info(f"Successfully loaded custom code for adapter '{adapter_name}'")
            return formatter_handler
            
        except Exception as e:
            logger.exception(f"Failed to load custom code for adapter '{adapter_name}'")
            raise

    def get_formatter_handler(self, adapter_name: str) -> Optional[CustomFormatterHandler]:
        """
        Retrieves the formatter handler for a specific adapter.
        
        Args:
            adapter_name: Unique identifier for the adapter
            
        Returns:
            CustomFormatterHandler if adapter has custom code, None otherwise
        """
        return self.code_registry.get(adapter_name)

    def unload_adapter_code(self, adapter_name: str) -> bool:
        """
        Removes adapter's custom code from registry.
        
        Args:
            adapter_name: Unique identifier for the adapter
            
        Returns:
            True if adapter was found and unloaded, False if not found
        """
        if adapter_name not in self.code_registry:
            logger.debug(f"Adapter '{adapter_name}' not found in code registry")
            return False
        
        logger.info(f"Unloading custom code for adapter '{adapter_name}'")
        del self.code_registry[adapter_name]
        
        return True
