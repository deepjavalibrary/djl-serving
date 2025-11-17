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
"""
Tests for adapter registration with custom formatters.

This test suite verifies:
- Registering adapters with custom input/output formatters
- Registering adapters without custom code (backward compatibility)
- Adapter metadata in registry
- Error handling for broken custom code
- Multiple adapter isolation
"""

import asyncio
import os
import tempfile
import shutil
import unittest
from unittest.mock import AsyncMock, MagicMock

from djl_python.inputs import Input


class TestAdapterRegistration(unittest.TestCase):
    """Tests for adapter registration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter_name = "test_adapter"
        self.adapter_alias = "test_adapter"
        self.test_adapters = {}

    def tearDown(self):
        """Clean up test fixtures"""
        import sys

        # Clean up any ns_* modules from sys.modules
        modules_to_remove = [
            m for m in sys.modules.keys() if m.startswith('ns_')
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        # Remove temp directories from sys.path
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)

        # Remove adapter directories from sys.path
        for adapter_name in self.test_adapters:
            adapter_dir = self.test_adapters[adapter_name]
            if adapter_dir in sys.path:
                sys.path.remove(adapter_dir)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_service(self):
        """Create a mock service with adapter management"""
        from djl_python.adapter_formatter_mixin import AdapterFormatterMixin

        class MockService(AdapterFormatterMixin):

            def __init__(self):
                AdapterFormatterMixin.__init__(self)

            async def add_lora(self, lora_name: str, lora_alias: str,
                               lora_path: str):
                return True

            async def remove_lora(self, lora_name: str, lora_alias: str):
                pass

            async def pin_lora(self, lora_name: str, lora_alias: str):
                pass

        return MockService()

    def _create_adapter_directory(self,
                                  adapter_name: str,
                                  input_formatter_code: str = None,
                                  output_formatter_code: str = None,
                                  broken: bool = False) -> str:
        """Create an adapter directory with custom code"""
        adapter_dir = os.path.join(self.temp_dir, adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create adapter_config.json
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            f.write(
                '{"base_model_name_or_path": "test-model", "peft_type": "LORA"}'
            )

        # Create model.py with custom formatters
        if input_formatter_code or output_formatter_code or broken:
            model_py_path = os.path.join(adapter_dir, "model.py")

            if broken:
                # Create broken code with syntax error
                model_py_content = '''
def input_formatter(func):
    func.is_input_formatter = True
    return func

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    # Syntax error - missing closing parenthesis
    result = dict(
    return result
'''
            else:
                # Create valid custom formatters
                model_py_content = '''
"""Custom formatters for {adapter_name}"""

def input_formatter(func):
    """Decorator to mark input formatter"""
    func.is_input_formatter = True
    return func

def output_formatter(func):
    """Decorator to mark output formatter"""
    func.is_output_formatter = True
    return func

'''.format(adapter_name=adapter_name)

                if input_formatter_code:
                    model_py_content += f"\n{input_formatter_code}\n"

                if output_formatter_code:
                    model_py_content += f"\n{output_formatter_code}\n"

            with open(model_py_path, 'w') as f:
                f.write(model_py_content)

        # Add adapter directory to sys.path
        import sys
        if adapter_dir not in sys.path:
            sys.path.insert(0, adapter_dir)

        self.test_adapters[adapter_name] = adapter_dir
        return adapter_dir

    def test_register_adapter_with_custom_formatters(self):
        """Test registering an adapter with custom input and output formatters"""
        input_formatter = '''
@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    """Add prefix to input"""
    if isinstance(decoded_payload, dict) and 'inputs' in decoded_payload:
        decoded_payload['inputs'] = f"[ADAPTER1] {decoded_payload['inputs']}"
    return decoded_payload
'''

        output_formatter = '''
@output_formatter
def custom_output_formatter(output, **kwargs):
    """Add suffix to output"""
    if isinstance(output, str):
        return f"{output} [PROCESSED_BY_ADAPTER1]"
    return output
'''

        adapter_dir = self._create_adapter_directory(
            "test_adapter_1",
            input_formatter_code=input_formatter,
            output_formatter_code=output_formatter)

        async def run_test():
            service = self._create_mock_service()

            # Load adapter custom code
            formatter_handler = service.load_adapter_custom_code(
                "test_adapter_1", adapter_dir)

            # Verify formatter handler was created
            self.assertIsNotNone(formatter_handler)
            self.assertTrue(hasattr(formatter_handler, 'input_formatter'))
            self.assertTrue(hasattr(formatter_handler, 'output_formatter'))

            # Test input formatter
            test_input = {"inputs": "Hello"}
            formatted_input = formatter_handler.input_formatter(test_input)
            self.assertEqual(formatted_input['inputs'], "[ADAPTER1] Hello")

            # Test output formatter
            test_output = "World"
            formatted_output = formatter_handler.output_formatter(test_output)
            self.assertEqual(formatted_output, "World [PROCESSED_BY_ADAPTER1]")

        asyncio.run(run_test())

    def test_register_adapter_without_custom_code(self):
        """Test registering an adapter without custom code (backward compatibility)"""
        adapter_dir = self._create_adapter_directory("test_adapter_no_code")

        async def run_test():
            service = self._create_mock_service()

            # Verify model.py doesn't exist
            model_py_path = os.path.join(adapter_dir, "model.py")
            self.assertFalse(os.path.exists(model_py_path))

            # Verify get_adapter_formatter_handler returns None
            formatter_handler = service.get_adapter_formatter_handler(
                "test_adapter_no_code")
            self.assertIsNone(formatter_handler)

        asyncio.run(run_test())

    def test_register_adapter_with_broken_custom_code(self):
        """Test that registering an adapter with broken custom code fails"""
        adapter_dir = self._create_adapter_directory("test_adapter_broken",
                                                     broken=True)

        async def run_test():
            from djl_python.custom_formatter_handling import CustomFormatterError

            service = self._create_mock_service()

            # Try to load broken adapter code - should raise exception
            with self.assertRaises(
                (CustomFormatterError, ValueError, SyntaxError, Exception)):
                service.load_adapter_custom_code("test_adapter_broken",
                                                 adapter_dir)

        asyncio.run(run_test())

    def test_adapter_metadata_with_custom_code(self):
        """Test that adapter registry includes custom code metadata"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create model.py with formatters
        model_py_content = '''
def input_formatter(func):
    func.is_input_formatter = True
    return func

def output_formatter(func):
    func.is_output_formatter = True
    return func

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    return decoded_payload

@output_formatter
def custom_output_formatter(output, **kwargs):
    return output
'''
        model_py_path = os.path.join(adapter_dir, "model.py")
        with open(model_py_path, 'w') as f:
            f.write(model_py_content)

        # Add to sys.path
        import sys
        if adapter_dir not in sys.path:
            sys.path.insert(0, adapter_dir)

        service = self._create_mock_service()

        # Create input for registration
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias
        inputs.properties["src"] = adapter_dir
        inputs.properties["preload"] = "true"
        inputs.properties["pin"] = "false"

        async def run_test():
            result = await service.register_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify adapter was registered
        self.assertIn(self.adapter_name, service.adapter_registry)

        # Verify custom code metadata is present
        registered_adapter = service.adapter_registry[self.adapter_name]
        self.assertEqual(registered_adapter.get_property("has_custom_code"),
                         "true")

        # Verify custom code is loaded
        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNotNone(formatter_handler)
        self.assertIsNotNone(formatter_handler.input_formatter)
        self.assertIsNotNone(formatter_handler.output_formatter)

        # Clean up sys.path
        if adapter_dir in sys.path:
            sys.path.remove(adapter_dir)

    def test_adapter_metadata_without_custom_code(self):
        """Test that adapter registry correctly indicates no custom code"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        service = self._create_mock_service()

        # Create input for registration
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias
        inputs.properties["src"] = adapter_dir
        inputs.properties["preload"] = "true"
        inputs.properties["pin"] = "false"

        async def run_test():
            result = await service.register_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify adapter was registered
        self.assertIn(self.adapter_name, service.adapter_registry)

        # Verify custom code metadata indicates no custom code
        registered_adapter = service.adapter_registry[self.adapter_name]
        self.assertEqual(registered_adapter.get_property("has_custom_code"),
                         "false")

        # Verify no custom code is loaded
        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNone(formatter_handler)

    def test_adapter_metadata_with_partial_formatters(self):
        """Test adapter with only input formatter"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create model.py with only input formatter
        model_py_content = '''
def input_formatter(func):
    func.is_input_formatter = True
    return func

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    return decoded_payload
'''
        model_py_path = os.path.join(adapter_dir, "model.py")
        with open(model_py_path, 'w') as f:
            f.write(model_py_content)

        # Add to sys.path
        import sys
        if adapter_dir not in sys.path:
            sys.path.insert(0, adapter_dir)

        service = self._create_mock_service()

        # Create input for registration
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias
        inputs.properties["src"] = adapter_dir
        inputs.properties["preload"] = "true"
        inputs.properties["pin"] = "false"

        async def run_test():
            result = await service.register_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify adapter was registered
        self.assertIn(self.adapter_name, service.adapter_registry)

        # Verify custom code is loaded with only input formatter
        registered_adapter = service.adapter_registry[self.adapter_name]
        self.assertEqual(registered_adapter.get_property("has_custom_code"),
                         "true")

        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNotNone(formatter_handler)
        self.assertIsNotNone(formatter_handler.input_formatter)
        self.assertIsNone(formatter_handler.output_formatter)

        # Clean up sys.path
        if adapter_dir in sys.path:
            sys.path.remove(adapter_dir)


if __name__ == '__main__':
    unittest.main()
