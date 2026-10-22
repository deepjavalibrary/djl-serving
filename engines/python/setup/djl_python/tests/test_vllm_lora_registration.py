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
"""
Tests for LoRA loading failures in VLLMHandler.add_lora.

The adapter tests in test_adapter_registration.py and test_adapter_lifecycle.py stub add_lora
to return True, so they cannot see either of the checks here. These drive the real
VLLMHandler.add_lora with a mocked engine, and the two mixin callers that answer a client.
"""

import asyncio
import json
import os
import shutil
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.pair_list import PairList


def _decode_output(output: Output) -> dict:
    raw = output.content.value_at(0)
    num_pairs = struct.unpack('>h', raw[:2])[0]
    offset = 2
    pairs = {}
    for _ in range(num_pairs):
        key_len = struct.unpack('>i', raw[offset:offset + 4])[0]
        offset += 4
        key = raw[offset:offset + key_len].decode('utf-8')
        offset += key_len
        val_len = struct.unpack('>i', raw[offset:offset + 4])[0]
        offset += 4
        val = raw[offset:offset + val_len].decode('utf-8')
        offset += val_len
        pairs[key] = val
    return pairs


class TestVllmAddLora(unittest.TestCase):
    """Tests for VLLMHandler.add_lora's two engine-safety checks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _adapter_dir(self, name: str, with_config: bool = True) -> str:
        """Creates an adapter source directory, with or without adapter_config.json."""
        adapter_dir = os.path.join(self.temp_dir, name)
        os.makedirs(adapter_dir, exist_ok=True)
        if with_config:
            with open(os.path.join(adapter_dir, "adapter_config.json"),
                      'w') as f:
                json.dump(
                    {
                        "base_model_name_or_path": "test-model",
                        "peft_type": "LORA"
                    }, f)
        return adapter_dir

    def _handler(self, add_lora_result=True):
        """A handler whose engine is mocked, so no adapter is really loaded."""
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler

        handler = VLLMHandler()
        handler.vllm_engine = MagicMock()
        if isinstance(add_lora_result, Exception):
            handler.vllm_engine.add_lora = AsyncMock(
                side_effect=add_lora_result)
        else:
            handler.vllm_engine.add_lora = AsyncMock(
                return_value=add_lora_result)
        handler.vllm_engine.remove_lora = AsyncMock(return_value=True)
        handler.model_registry = SimpleNamespace(lora_requests={})
        return handler

    def _registries(self, handler):
        return handler.model_registry.lora_requests, handler.lora_requests

    @patch('djl_python.lmi_vllm.vllm_async_service.create_lora_request')
    def test_missing_config_does_not_reach_the_engine(self, mock_create):
        handler = self._handler()
        path = self._adapter_dir("broken", with_config=False)

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(handler.add_lora("broken", "broken", path))

        self.assertIn("adapter_config.json", str(ctx.exception))
        # The engine core dies inside add_lora on a missing config, so not calling it is the
        # whole point of the check: a return-value check alone cannot save the model.
        handler.vllm_engine.add_lora.assert_not_called()
        mock_create.assert_not_called()
        registry, requests = self._registries(handler)
        self.assertEqual(registry, {})
        self.assertEqual(requests, {})

    @patch('djl_python.lmi_vllm.vllm_async_service.create_lora_request')
    def test_engine_rejection_leaves_no_registry_state(self, mock_create):
        handler = self._handler(add_lora_result=False)
        mock_create.return_value = SimpleNamespace(lora_name="rejected")
        path = self._adapter_dir("rejected")

        with self.assertRaises(ValueError):
            asyncio.run(handler.add_lora("rejected", "rejected", path))

        handler.vllm_engine.add_lora.assert_awaited_once()
        # A false return used to take the success path, leaving a lora_int_id the engine never
        # loaded reachable by later requests.
        registry, requests = self._registries(handler)
        self.assertEqual(registry, {})
        self.assertEqual(requests, {})

    @patch('djl_python.lmi_vllm.vllm_async_service.create_lora_request')
    def test_engine_exception_leaves_no_registry_state(self, mock_create):
        handler = self._handler(add_lora_result=RuntimeError("engine dead"))
        mock_create.return_value = SimpleNamespace(lora_name="exploded")
        path = self._adapter_dir("exploded")

        with self.assertRaises(RuntimeError):
            asyncio.run(handler.add_lora("exploded", "exploded", path))

        registry, requests = self._registries(handler)
        self.assertEqual(registry, {})
        self.assertEqual(requests, {})

    @patch('djl_python.lmi_vllm.vllm_async_service.create_lora_request')
    def test_successful_load_registers_after_the_engine_holds_it(
            self, mock_create):
        handler = self._handler()
        lora_request = SimpleNamespace(lora_name="good")
        mock_create.return_value = lora_request
        path = self._adapter_dir("good")

        seen = {}

        async def record(_request):
            # Captured while the engine call is in flight: registering before it is what left a
            # dead adapter resolvable.
            registry, requests = self._registries(handler)
            seen["registry"] = dict(registry)
            seen["requests"] = dict(requests)
            return True

        handler.vllm_engine.add_lora = AsyncMock(side_effect=record)

        result = asyncio.run(handler.add_lora("good", "good", path))

        self.assertTrue(result)
        self.assertEqual(seen["registry"], {})
        self.assertEqual(seen["requests"], {})
        registry, requests = self._registries(handler)
        self.assertIs(registry["good"], lora_request)
        self.assertIs(requests["good"], lora_request)


class TestAdapterCallersReportLoadFailure(unittest.TestCase):
    """The two mixin callers must answer 424 rather than OK when the load fails."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _handler(self):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler

        handler = VLLMHandler()
        handler.vllm_engine = MagicMock()
        handler.vllm_engine.add_lora = AsyncMock(return_value=True)
        handler.vllm_engine.remove_lora = AsyncMock(return_value=True)
        handler.model_registry = SimpleNamespace(lora_requests={})
        return handler

    def _broken_adapter_dir(self) -> str:
        # Exists, so register_adapter's own path check passes and the failure comes from the
        # load rather than from the argument validation above it.
        adapter_dir = os.path.join(self.temp_dir, "broken")
        os.makedirs(adapter_dir, exist_ok=True)
        return adapter_dir

    @staticmethod
    def _input(content=None, **properties) -> Input:
        # name, alias and src are read with get_property; preload and pin are read from the
        # content with contains_key and get_as_string.
        inp = Input()
        inp.properties.update(properties)
        inp.content = PairList()
        for key, value in (content or {}).items():
            inp.content.add(key=key, value=value.encode('utf-8'))
        return inp

    def test_register_adapter_returns_424(self):
        handler = self._handler()
        path = self._broken_adapter_dir()

        outputs = asyncio.run(
            handler.register_adapter(self._input(name="broken", src=path)))

        # binary_encode writes each value with str(), so the code arrives as text.
        decoded = _decode_output(outputs)
        self.assertEqual(decoded["code"], "424")
        self.assertIn("adapter_config.json", decoded["error"])
        self.assertNotIn("broken", handler.adapter_registry)
        self.assertEqual(handler.model_registry.lora_requests, {})
        self.assertEqual(handler.lora_requests, {})

    def test_update_adapter_turning_preload_on_returns_424(self):
        handler = self._handler()
        path = self._broken_adapter_dir()
        # Registered with preload=false, so no load was attempted and the adapter is in the
        # registry. Turning preload on is the second door into add_lora.
        registered = self._input(content={"preload": "false"},
                                 name="broken",
                                 src=path)
        outputs = asyncio.run(handler.register_adapter(registered))
        self.assertNotIn("code", _decode_output(outputs))

        outputs = asyncio.run(
            handler.update_adapter(
                self._input(content={"preload": "true"},
                            name="broken",
                            src=path)))

        decoded = _decode_output(outputs)
        self.assertEqual(decoded["code"], "424")
        self.assertIn("adapter_config.json", decoded["error"])
        self.assertEqual(handler.model_registry.lora_requests, {})
        self.assertEqual(handler.lora_requests, {})


if __name__ == '__main__':
    unittest.main()
