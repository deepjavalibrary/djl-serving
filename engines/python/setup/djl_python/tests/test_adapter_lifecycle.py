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
Tests for adapter lifecycle operations.

This test suite verifies:
- Adapter deletion with custom code cleanup
- Multiple adapter isolation
- Inference with custom formatters
- Error handling during deletion
"""

import asyncio
import os
import tempfile
import shutil
import unittest
from unittest.mock import AsyncMock, MagicMock

from djl_python.inputs import Input


class TestAdapterLifecycle(unittest.TestCase):
    """Tests for adapter lifecycle operations"""

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
                self.remove_lora_called = False

            async def add_lora(self, lora_name: str, lora_alias: str,
                               lora_path: str):
                return True

            async def remove_lora(self, lora_name: str, lora_alias: str):
                self.remove_lora_called = True

            async def pin_lora(self, lora_name: str, lora_alias: str):
                pass

        return MockService()

    def _create_adapter_directory(self,
                                  adapter_name: str,
                                  input_formatter_code: str = None,
                                  output_formatter_code: str = None) -> str:
        """Create an adapter directory with custom code"""
        adapter_dir = os.path.join(self.temp_dir, adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create model.py with custom formatters
        if input_formatter_code or output_formatter_code:
            model_py_path = os.path.join(adapter_dir, "model.py")

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

    def test_delete_adapter_with_custom_code(self):
        """Test deleting an adapter with custom code"""
        input_formatter = '''
@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    return decoded_payload
'''

        adapter_dir = self._create_adapter_directory(
            "test_adapter_delete", input_formatter_code=input_formatter)

        async def run_test():
            service = self._create_mock_service()

            # Load adapter
            service.load_adapter_custom_code("test_adapter_delete",
                                             adapter_dir)

            # Verify it's loaded
            formatter_handler = service.get_adapter_formatter_handler(
                "test_adapter_delete")
            self.assertIsNotNone(formatter_handler)

            # Unload adapter
            result = service.unload_adapter_custom_code("test_adapter_delete")
            self.assertTrue(result)

            # Verify it's removed
            formatter_handler = service.get_adapter_formatter_handler(
                "test_adapter_delete")
            self.assertIsNone(formatter_handler)

        asyncio.run(run_test())

    def test_unregister_adapter_with_custom_code(self):
        """Test that unregister_adapter removes both weights and custom code"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create model.py
        model_py_content = '''
def input_formatter(func):
    func.is_input_formatter = True
    return func

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    return decoded_payload
'''
        model_py_path = os.path.join(adapter_dir, "model.py")
        with open(model_py_path, "w") as f:
            f.write(model_py_content)

        # Add to sys.path
        import sys
        if adapter_dir not in sys.path:
            sys.path.insert(0, adapter_dir)

        service = self._create_mock_service()

        # Register adapter first
        register_inputs = Input()
        register_inputs.properties["name"] = self.adapter_name
        register_inputs.properties["alias"] = self.adapter_alias
        register_inputs.properties["src"] = adapter_dir
        register_inputs.properties["preload"] = "true"
        register_inputs.properties["pin"] = "false"

        async def run_register():
            result = await service.register_adapter(register_inputs)
            return result

        asyncio.run(run_register())

        # Verify adapter was registered with custom code
        self.assertIn(self.adapter_name, service.adapter_registry)
        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNotNone(formatter_handler)

        # Create input for unregister
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias

        async def run_test():
            result = await service.unregister_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify custom code was unloaded
        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNone(formatter_handler)

        # Verify adapter was removed from registry
        self.assertNotIn(self.adapter_name, service.adapter_registry)

        # Verify remove_lora was called
        self.assertTrue(service.remove_lora_called)

        # Clean up sys.path
        if adapter_dir in sys.path:
            sys.path.remove(adapter_dir)

    def test_unregister_adapter_without_custom_code(self):
        """Test that unregister_adapter works for adapters without custom code"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        service = self._create_mock_service()

        # Register adapter first
        register_inputs = Input()
        register_inputs.properties["name"] = self.adapter_name
        register_inputs.properties["alias"] = self.adapter_alias
        register_inputs.properties["src"] = adapter_dir
        register_inputs.properties["preload"] = "true"
        register_inputs.properties["pin"] = "false"

        async def run_register():
            result = await service.register_adapter(register_inputs)
            return result

        asyncio.run(run_register())

        # Verify adapter was registered without custom code
        self.assertIn(self.adapter_name, service.adapter_registry)
        formatter_handler = service.get_adapter_formatter_handler(
            self.adapter_name)
        self.assertIsNone(formatter_handler)

        # Create input for unregister
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias

        async def run_test():
            result = await service.unregister_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify adapter was removed from registry
        self.assertNotIn(self.adapter_name, service.adapter_registry)

        # Verify remove_lora was called
        self.assertTrue(service.remove_lora_called)

    def test_unregister_adapter_partial_failure(self):
        """Test that partial deletion is reported correctly"""
        adapter_dir = os.path.join(self.temp_dir, self.adapter_name)
        os.makedirs(adapter_dir)

        # Create __init__.py
        init_path = os.path.join(adapter_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # Create model.py
        model_py_content = '''
def input_formatter(func):
    func.is_input_formatter = True
    return func

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    return decoded_payload
'''
        model_py_path = os.path.join(adapter_dir, "model.py")
        with open(model_py_path, "w") as f:
            f.write(model_py_content)

        # Add to sys.path
        import sys
        if adapter_dir not in sys.path:
            sys.path.insert(0, adapter_dir)

        # Create mock service with failing remove_lora
        from djl_python.adapter_formatter_mixin import AdapterFormatterMixin

        class MockServiceWithFailure(AdapterFormatterMixin):

            def __init__(self):
                AdapterFormatterMixin.__init__(self)

            async def add_lora(self, lora_name: str, lora_alias: str,
                               lora_path: str):
                return True

            async def remove_lora(self, lora_name: str, lora_alias: str):
                raise Exception("Failed to remove weights")

            async def pin_lora(self, lora_name: str, lora_alias: str):
                pass

        service = MockServiceWithFailure()

        # Register adapter first
        register_inputs = Input()
        register_inputs.properties["name"] = self.adapter_name
        register_inputs.properties["alias"] = self.adapter_alias
        register_inputs.properties["src"] = adapter_dir
        register_inputs.properties["preload"] = "true"
        register_inputs.properties["pin"] = "false"

        async def run_register():
            result = await service.register_adapter(register_inputs)
            return result

        asyncio.run(run_register())

        # Verify adapter was registered with custom code
        self.assertIn(self.adapter_name, service.adapter_registry)

        # Create input for unregister
        inputs = Input()
        inputs.properties["name"] = self.adapter_name
        inputs.properties["alias"] = self.adapter_alias

        async def run_test():
            result = await service.unregister_adapter(inputs)
            return result

        result = asyncio.run(run_test())

        # Verify the result contains error information
        self.assertGreater(result.content.size(), 0)

        # Verify adapter was NOT removed from registry (due to failure)
        self.assertIn(self.adapter_name, service.adapter_registry)

        # Clean up sys.path
        if adapter_dir in sys.path:
            sys.path.remove(adapter_dir)

    def test_multiple_adapters_isolation(self):
        """Test that multiple adapters with different custom code are isolated"""
        # Create adapter 1
        input_formatter_1 = '''
@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    if isinstance(decoded_payload, dict) and 'inputs' in decoded_payload:
        decoded_payload['inputs'] = f"[A1] {decoded_payload['inputs']}"
    return decoded_payload
'''

        output_formatter_1 = '''
@output_formatter
def custom_output_formatter(output, **kwargs):
    if isinstance(output, str):
        return f"{output} [A1_OUT]"
    return output
'''

        # Create adapter 2
        input_formatter_2 = '''
@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    if isinstance(decoded_payload, dict) and 'inputs' in decoded_payload:
        decoded_payload['inputs'] = f">>> {decoded_payload['inputs']} <<<"
    return decoded_payload
'''

        output_formatter_2 = '''
@output_formatter
def custom_output_formatter(output, **kwargs):
    if isinstance(output, str):
        return f"*** {output} ***"
    return output
'''

        adapter_dir_1 = self._create_adapter_directory(
            "adapter_1",
            input_formatter_code=input_formatter_1,
            output_formatter_code=output_formatter_1)

        adapter_dir_2 = self._create_adapter_directory(
            "adapter_2",
            input_formatter_code=input_formatter_2,
            output_formatter_code=output_formatter_2)

        async def run_test():
            service = self._create_mock_service()

            # Load both adapters
            handler_1 = service.load_adapter_custom_code(
                "adapter_1", adapter_dir_1)
            handler_2 = service.load_adapter_custom_code(
                "adapter_2", adapter_dir_2)

            # Test adapter 1 formatters
            test_input_1 = {"inputs": "Test"}
            formatted_1 = handler_1.input_formatter(test_input_1)
            self.assertEqual(formatted_1['inputs'], "[A1] Test")

            output_1 = handler_1.output_formatter("Result")
            self.assertEqual(output_1, "Result [A1_OUT]")

            # Test adapter 2 formatters
            test_input_2 = {"inputs": "Test"}
            formatted_2 = handler_2.input_formatter(test_input_2)
            self.assertEqual(formatted_2['inputs'], ">>> Test <<<")

            output_2 = handler_2.output_formatter("Result")
            self.assertEqual(output_2, "*** Result ***")

            # Verify both adapters are in the registry
            self.assertIsNotNone(
                service.get_adapter_formatter_handler("adapter_1"))
            self.assertIsNotNone(
                service.get_adapter_formatter_handler("adapter_2"))

            # Verify formatters remain isolated
            test_input_1_again = {"inputs": "Test2"}
            formatted_1_again = handler_1.input_formatter(test_input_1_again)
            self.assertEqual(formatted_1_again['inputs'], "[A1] Test2")

            test_input_2_again = {"inputs": "Test2"}
            formatted_2_again = handler_2.input_formatter(test_input_2_again)
            self.assertEqual(formatted_2_again['inputs'], ">>> Test2 <<<")

        asyncio.run(run_test())

    def test_inference_with_adapter_custom_formatter(self):
        """Test end-to-end inference with adapter custom formatters"""
        input_formatter = '''
@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    """Transform input for adapter"""
    if isinstance(decoded_payload, dict):
        if 'inputs' in decoded_payload:
            decoded_payload['inputs'] = f"CUSTOM: {decoded_payload['inputs']}"
        elif 'prompt' in decoded_payload:
            decoded_payload['prompt'] = f"CUSTOM: {decoded_payload['prompt']}"
    return decoded_payload
'''

        output_formatter = '''
@output_formatter
def custom_output_formatter(output, **kwargs):
    """Transform output for adapter"""
    if isinstance(output, dict) and 'generated_text' in output:
        output['generated_text'] = f"{output['generated_text']} [CUSTOM]"
    elif isinstance(output, str):
        output = f"{output} [CUSTOM]"
    return output
'''

        adapter_dir = self._create_adapter_directory(
            "inference_adapter",
            input_formatter_code=input_formatter,
            output_formatter_code=output_formatter)

        async def run_test():
            service = self._create_mock_service()

            # Load adapter
            formatter_handler = service.load_adapter_custom_code(
                "inference_adapter", adapter_dir)

            # Simulate input formatting
            input_data = {"inputs": "Hello world"}
            formatted_input = formatter_handler.input_formatter(input_data)

            # Verify input was transformed
            self.assertEqual(formatted_input['inputs'], "CUSTOM: Hello world")

            # Simulate output formatting
            output_data = {"generated_text": "Response"}
            formatted_output = formatter_handler.output_formatter(output_data)

            # Verify output was transformed
            self.assertEqual(formatted_output['generated_text'],
                             "Response [CUSTOM]")

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
