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
import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

from vllm.exceptions import VLLMValidationError

from djl_python.lmi_vllm import vllm_async_service as service
from djl_python.lmi_vllm.request_response_utils import (ProcessedRequest,
                                                        name_lmi_options)
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.test_model import decode_encoded_output_binary


class TestTrustRequestMmKwargsProperty(unittest.TestCase):

    BASE = {"engine": "Python", "model_id": "some_model"}

    def test_default_is_false(self):
        self.assertFalse(VllmRbProperties(**self.BASE).trust_request_mm_kwargs)

    def test_parses_the_serving_properties_string(self):
        props = VllmRbProperties(**{
            **self.BASE, "trust_request_mm_kwargs": "true"
        })

        self.assertTrue(props.trust_request_mm_kwargs)

    def test_is_not_passed_to_the_vllm_engine(self):
        # It is a serving layer switch, not an EngineArgs field
        props = VllmRbProperties(**{
            **self.BASE, "trust_request_mm_kwargs": "true"
        })

        self.assertNotIn("trust_request_mm_kwargs",
                         props.get_additional_vllm_engine_args())
        self.assertNotIn("trust_request_mm_kwargs",
                         props.generate_vllm_engine_arg_dict({}))


def _initialize(properties):
    """Runs VLLMHandler.initialize against mocked vLLM serving classes."""
    handler = service.VLLMHandler()
    names = ("AsyncLLMEngine", "OpenAIServingModels", "OnlineRenderer",
             "OpenAIServingCompletion", "OpenAIServingChat",
             "ServingEmbedding", "SessionManager")
    with tempfile.TemporaryDirectory() as model_dir, \
            ExitStack() as stack:
        properties = {
            "engine": "Python",
            "model_id": "some_model",
            "model_dir": model_dir,
            **properties
        }
        mocks = {
            name: stack.enter_context(patch.object(service, name))
            for name in names
        }
        stack.enter_context(
            patch.object(service, "load_chat_template", return_value=None))
        stack.enter_context(
            patch.object(VllmRbProperties,
                         "get_engine_args",
                         return_value=MagicMock(served_model_name=None,
                                                model="some_model")))
        asyncio.run(handler.initialize(properties))
    return handler, mocks


class TestInitialize(unittest.TestCase):

    def test_online_renderer_rejects_request_mm_kwargs_by_default(self):
        _, mocks = _initialize({})

        kwargs = mocks["OnlineRenderer"].call_args.kwargs
        self.assertIs(False, kwargs["trust_request_mm_kwargs"])

    def test_online_renderer_trusts_request_mm_kwargs_when_configured(self):
        _, mocks = _initialize({"trust_request_mm_kwargs": "true"})

        kwargs = mocks["OnlineRenderer"].call_args.kwargs
        self.assertIs(True, kwargs["trust_request_mm_kwargs"])

    def test_embedding_service_follows_the_same_property(self):
        for value in (False, True):
            _, mocks = _initialize({
                "task":
                "text-embedding",
                "trust_request_mm_kwargs":
                str(value).lower()
            })

            config = mocks["ServingEmbedding"].call_args.kwargs[
                "chat_template_config"]
            self.assertIs(value, config.trust_request_mm_kwargs)


class TestRejectedRequest(unittest.TestCase):
    """vLLM raises VLLMValidationError for per-request multimodal kwargs without
    trust. Its web server answers 400 with the message."""

    MESSAGE = (
        "Per-request mm_processor_kwargs are disabled by default. Start "
        "the server with --trust-request-mm-kwargs only when clients "
        "are trusted.")

    def _inference(self, error):
        handler = service.VLLMHandler()
        processed = ProcessedRequest(MagicMock(), AsyncMock(side_effect=error),
                                     MagicMock(), MagicMock(), False, False)
        with patch.object(handler, "check_health", AsyncMock()), \
                patch.object(handler, "preprocess_request",
                             return_value=processed):
            return asyncio.run(handler.inference(MagicMock()))

    def test_rejected_mm_kwargs_are_a_400_that_names_the_lmi_option(self):
        for parameter in ("mm_processor_kwargs", "media_io_kwargs"):
            with self.subTest(parameter):
                output = self._inference(
                    VLLMValidationError(self.MESSAGE, parameter=parameter))

                decoded = decode_encoded_output_binary(
                    output.content.value_at(0))

                self.assertEqual("400", decoded["code"])
                self.assertEqual("True", decoded["last"])
                self.assertIn("option.trust_request_mm_kwargs",
                              decoded["error"])
                self.assertNotIn("--trust-request-mm-kwargs", decoded["error"])
                self.assertIn(f"parameter={parameter}", decoded["error"])

    def test_other_errors_are_left_to_the_generic_handler(self):
        for error in (VLLMValidationError("too long", parameter="max_tokens"),
                      RuntimeError("engine failed")):
            with self.subTest(type(error).__name__), \
                    self.assertRaises(type(error)):
                self._inference(error)

    def test_only_the_flags_of_lmi_options_are_replaced(self):
        self.assertEqual(
            "Start with option.trust_request_mm_kwargs, not --other-flag",
            name_lmi_options(
                "Start with --trust-request-mm-kwargs, not --other-flag"))


if __name__ == '__main__':
    unittest.main()
