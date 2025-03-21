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
import types
from typing import Optional, Tuple, Callable, Union, AsyncGenerator

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath

from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import decode

logger = logging.getLogger(__name__)


# TODO: support handle returning AsyncGenerator directly so that
# users don't have to work directly with the socket object this way
async def handle_streaming_response(response: AsyncGenerator[str, None],
                                    properties: dict, cl_socket) -> Output:
    async for chunk in response:
        # TODO: support LMI output schema
        output = Output()
        for k, v in properties.items():
            output.add_property(k, v)
        trimmed_chunk = chunk[6:]
        if trimmed_chunk == "[DONE]\n\n":
            data = ""
            last = True
        else:
            data = trimmed_chunk
            last = False
        resp = {"data": data, "last": last}
        output.add(Output.binary_encode(resp))
        if last:
            return output
        output.send(cl_socket)


def convert_lmi_schema_to_completion_request(
        payload: dict) -> CompletionRequest:
    completion_dict = {
        "prompt": payload.get("inputs"),
        "stream": payload.get("stream", False),
        "model": payload.get("model"),
        "max_tokens": payload.pop("max_new_tokens",
                                  payload.pop("max_tokens", 30))
    }
    # TODO: Sampling Param conversion
    return CompletionRequest(**completion_dict)


class VLLMHandler:

    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.chat_completion_service = None
        self.completion_service = None
        self.model_registry = None
        self.hf_configs = None
        self.vllm_engine_args = None
        self.vllm_properties = None
        self.model_name = None
        self.initialized = False

    async def initialize(self, properties: dict):
        self.hf_configs = HuggingFaceProperties(**properties)
        self.vllm_properties = VllmRbProperties(**properties)
        self.vllm_engine_args = self.vllm_properties.get_engine_args(
            async_engine=True)
        self.vllm_engine = AsyncLLMEngine.from_engine_args(
            self.vllm_engine_args)
        model_config = await self.vllm_engine.get_model_config()

        model_names = self.vllm_engine_args.served_model_name or self.vllm_engine_args.model
        if not isinstance(model_names, list):
            model_names = [model_names]
        # Users can provide multiple names that refer to the same model
        base_model_paths = [
            BaseModelPath(model_name, self.vllm_engine_args.model)
            for model_name in model_names
        ]
        # Use the first model name as default.
        # This is needed to be backwards compatible since LMI never required the model name in payload
        self.model_name = model_names[0]
        self.model_registry = OpenAIServingModels(
            self.vllm_engine,
            model_config,
            base_model_paths,
        )
        self.completion_service = OpenAIServingCompletion(
            self.vllm_engine,
            model_config,
            self.model_registry,
            request_logger=None,
        )

        self.chat_completion_service = OpenAIServingChat(
            self.vllm_engine,
            model_config,
            self.model_registry,
            "assistant",
            request_logger=None,
            chat_template=self.vllm_properties.chat_template,
            chat_template_content_format=self.vllm_properties.
            chat_template_content_format,
            enable_auto_tools=self.vllm_properties.enable_auto_tool_choice,
            tool_parser=self.vllm_properties.tool_call_parser,
            enable_reasoning=self.vllm_properties.enable_reasoning,
            reasoning_parser=self.vllm_properties.reasoning_parser,
        )
        self.initialized = True

    def preprocess_request(
        self, inputs: Input
    ) -> Tuple[Union[CompletionRequest, ChatCompletionRequest], Callable]:
        batch = inputs.get_batches()
        assert len(batch) == 1, "only one request per batch allowed"
        raw_request = batch[0]
        content_type = raw_request.get_property("Content-Type")
        decoded_payload = decode(raw_request, content_type)
        if "model" not in decoded_payload:
            decoded_payload["model"] = self.model_name
        if "prompt" in decoded_payload:
            vllm_request = CompletionRequest(**decoded_payload)
            vllm_invoke_function = self.completion_service.create_completion
        elif "inputs" in decoded_payload:
            vllm_request = convert_lmi_schema_to_completion_request(
                decoded_payload)
            vllm_invoke_function = self.completion_service.create_completion
        elif "messages" in decoded_payload:
            vllm_request = ChatCompletionRequest(**decoded_payload)
            vllm_invoke_function = self.chat_completion_service.create_chat_completion
        else:
            raise RuntimeError(
                "invalid payload. must contain prompt, inputs, or messages")
        return vllm_request, vllm_invoke_function

    async def inference(self, inputs: Input, cl_socket) -> Output:
        properties = inputs.get_properties()
        try:
            request, invoke_call = self.preprocess_request(inputs)
        except Exception as e:
            logger.exception("Input parsing failed")
            output = Output()
            for k, v in properties.items():
                output.add_property(k, str(v))
            output.error(str(e), 424, "Input Parsing failed")
            return output

        response = await invoke_call(request)
        if isinstance(response, types.AsyncGeneratorType):
            return await handle_streaming_response(response, properties,
                                                   cl_socket)

        output = Output()
        response_dict = {"data": response.model_dump_json(), "last": True}
        if isinstance(response, ErrorResponse):
            response_dict["code"] = response.code
            response_dict["error"] = response.message
        for k, v in properties.items():
            output.add_property(k, str(v))
        output.add(Output.binary_encode(response_dict))
        return output


service = VLLMHandler()


async def handle(inputs: Input, cl_socket) -> Optional[Output]:
    if not service.initialized:
        await service.initialize(inputs.get_properties())
        logger.info("vllm service initialized")
    if inputs.is_empty():
        return None

    outputs = await service.inference(inputs, cl_socket)
    return outputs
