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
import types
from typing import Optional, Union, AsyncGenerator

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.utils import kill_process_tree

from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import decode
from djl_python.async_utils import handle_streaming_response, create_non_stream_output
from djl_python.custom_formatter_handling import CustomFormatterHandler, CustomFormatterError

from .request_response_utils import (
    ProcessedRequest,
    vllm_stream_output_formatter,
    vllm_non_stream_output_formatter,
    convert_lmi_schema_to_completion_request,
    lmi_with_details_stream_output_formatter,
    lmi_stream_output_formatter,
    lmi_with_details_non_stream_output_formatter,
    lmi_non_stream_output_formatter,
)

logger = logging.getLogger(__name__)


class VLLMHandler(CustomFormatterHandler):

    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.tokenizer = None
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

        # Load formatters
        model_dir = properties.get("model_dir", ".")
        try:
            self.load_formatters(model_dir)
        except CustomFormatterError as e:
            logger.error(
                f"Failed to initialize due to custom formatter error: {e}")
            raise

        self.vllm_engine_args = self.vllm_properties.get_engine_args(
            async_engine=True)
        self.vllm_engine = AsyncLLMEngine.from_engine_args(
            self.vllm_engine_args)
        self.tokenizer = await self.vllm_engine.get_tokenizer()
        model_config = await self.vllm_engine.get_model_config()

        model_names = self.vllm_engine_args.served_model_name or "lmi"
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
            reasoning_parser=self.vllm_properties.reasoning_parser,
        )
        self.initialized = True

    def preprocess_request(self, inputs: Input) -> ProcessedRequest:
        batch = inputs.get_batches()
        assert len(batch) == 1, "only one request per batch allowed"
        raw_request = batch[0]
        content_type = raw_request.get_property("Content-Type")
        decoded_payload = decode(raw_request, content_type)

        # Apply input formatter
        decoded_payload = self.apply_input_formatter(decoded_payload,
                                                     tokenizer=self.tokenizer)

        # For TGI streaming responses, the last chunk requires the full generated text to be provided.
        # Streaming completion responses only return deltas, so we need to accumulate chunks and construct
        # The full generation at the end... TODO is there a better way?
        accumulate_chunks = False
        include_prompt = False
        # completions/chat completions require model in the payload
        if "model" not in decoded_payload:
            decoded_payload["model"] = self.model_name
        # completions request
        if "prompt" in decoded_payload:
            vllm_request = CompletionRequest(**decoded_payload)
            vllm_invoke_function = self.completion_service.create_completion
            non_stream_output_formatter = vllm_non_stream_output_formatter
            stream_output_formatter = vllm_stream_output_formatter
        # TGI request gets mapped to completions
        elif "inputs" in decoded_payload:
            vllm_request, include_details, include_prompt = convert_lmi_schema_to_completion_request(
                decoded_payload)
            vllm_invoke_function = self.completion_service.create_completion
            non_stream_output_formatter = lmi_with_details_non_stream_output_formatter if include_details else lmi_non_stream_output_formatter
            stream_output_formatter = lmi_with_details_stream_output_formatter if include_details else lmi_stream_output_formatter
            accumulate_chunks = True
        # chat completions request
        elif "messages" in decoded_payload:
            vllm_request = ChatCompletionRequest(**decoded_payload)
            vllm_invoke_function = self.chat_completion_service.create_chat_completion
            non_stream_output_formatter = vllm_non_stream_output_formatter
            stream_output_formatter = vllm_stream_output_formatter
        else:
            raise RuntimeError(
                "invalid payload. must contain prompt, inputs, or messages")
        processed_request = ProcessedRequest(
            vllm_request,
            vllm_invoke_function,
            non_stream_output_formatter,
            stream_output_formatter,
            accumulate_chunks,
            include_prompt,
        )
        return processed_request

    async def check_health(self):
        try:
            await self.vllm_engine.check_health()
        except Exception as e:
            logger.fatal("vLLM engine is dead, terminating process")
            kill_process_tree(os.getpid())

    async def inference(
            self,
            inputs: Input) -> Union[Output, AsyncGenerator[Output, None]]:
        await self.check_health()
        try:
            processed_request = self.preprocess_request(inputs)
        except CustomFormatterError as e:
            logger.exception("Custom formatter failed")
            output = create_non_stream_output(
                "", error=f"Custom formatter failed: {str(e)}", code=424)
            return output
        except Exception as e:
            logger.exception("Input parsing failed")
            output = create_non_stream_output(
                "", error=f"Input parsing failed: {str(e)}", code=424)
            return output

        response = await processed_request.inference_invoker(
            processed_request.vllm_request)

        if isinstance(response, types.AsyncGeneratorType):
            # Apply custom formatter to streaming response
            response = self.apply_output_formatter_streaming_raw(response)

            return handle_streaming_response(
                response,
                processed_request.stream_output_formatter,
                request=processed_request.vllm_request,
                accumulate_chunks=processed_request.accumulate_chunks,
                include_prompt=processed_request.include_prompt,
                tokenizer=self.tokenizer,
            )

        # Apply custom output formatter to non-streaming response
        response = self.apply_output_formatter(response)

        return processed_request.non_stream_output_formatter(
            response,
            request=processed_request.vllm_request,
            tokenizer=self.tokenizer,
        )


service = VLLMHandler()


async def handle(
        inputs: Input
) -> Optional[Union[Output, AsyncGenerator[Output, None]]]:
    if not service.initialized:
        await service.initialize(inputs.get_properties())
        logger.info("vllm service initialized")
    if inputs.is_empty():
        return None

    outputs = await service.inference(inputs)
    return outputs
