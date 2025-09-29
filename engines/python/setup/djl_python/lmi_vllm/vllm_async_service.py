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
from vllm.utils import kill_process_tree, AtomicCounter

from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import decode
from djl_python.async_utils import handle_streaming_response, create_non_stream_output
from djl_python.custom_formatter_handling import CustomFormatterHandler, CustomFormatterError
from djl_python.rolling_batch.rolling_batch_vllm_utils import create_lora_request, get_lora_request
from djl_python.input_parser import SAGEMAKER_ADAPTER_IDENTIFIER_HEADER

from djl_python.lmi_vllm.request_response_utils import (
    ProcessedRequest,
    vllm_stream_output_formatter,
    vllm_non_stream_output_formatter,
    convert_lmi_schema_to_completion_request,
    lmi_with_details_stream_output_formatter,
    lmi_stream_output_formatter,
    lmi_with_details_non_stream_output_formatter,
    lmi_non_stream_output_formatter,
)
from djl_python.session_manager import SessionManager
from djl_python.session_utils import (create_session, close_session,
                                      get_session,
                                      session_non_stream_output_formatter)

logger = logging.getLogger(__name__)

SESSION_REQUESTS = {"NEW_SESSION": create_session, "CLOSE": close_session}


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
        self.adapter_registry = {}
        self.lora_id_counter = AtomicCounter(0)
        self.lora_requests = {}

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
        if properties.get("enable_stateful_sessions", "true") == "true":
            self.session_manager: SessionManager = SessionManager(properties)
        self.initialized = True

    def preprocess_request(self, inputs: Input) -> ProcessedRequest:
        batch = inputs.get_batches()
        assert len(batch) == 1, "only one request per batch allowed"
        raw_request = batch[0]
        session = get_session(self.session_manager, raw_request)
        content_type = raw_request.get_property("Content-Type")
        decoded_payload = decode(raw_request, content_type)

        adapter_name = self._extract_lora_adapter(raw_request, decoded_payload)

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

        lora_request = None
        if adapter_name:
            if adapter_name not in self.lora_requests:
                raise ValueError(
                    f"LoRA adapter {adapter_name} not found in registry. Available adapters: {list(self.lora_requests.keys())}"
                )
            lora_request = get_lora_request(adapter_name, self.lora_requests)
            logging.info(
                f"Using LoRA request: {lora_request.lora_name} (ID: {lora_request.lora_int_id})"
            )

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
        elif "requestType" in decoded_payload:
            request_type = decoded_payload["requestType"]
            if request_type not in SESSION_REQUESTS.keys():
                raise RuntimeError(
                    f"invalid payload. request type must be one of {SESSION_REQUESTS.keys()}"
                )
            if self.session_manager is None:
                raise RuntimeError(
                    f"invalid payload. stateful sessions not enabled, {request_type} not supported"
                )
            vllm_request = self.session_manager, inputs
            vllm_invoke_function = SESSION_REQUESTS[request_type]
            non_stream_output_formatter = session_non_stream_output_formatter
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
        processed_request.lora_request = lora_request
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

        if processed_request.lora_request:
            original_add_request = self.vllm_engine.add_request

            async def add_request_with_lora(*args, **kwargs):
                kwargs['lora_request'] = processed_request.lora_request
                return await original_add_request(*args, **kwargs)

            self.vllm_engine.add_request = add_request_with_lora

        try:
            response = await processed_request.inference_invoker(
                processed_request.vllm_request)
        finally:
            if processed_request.lora_request:
                self.vllm_engine.add_request = original_add_request

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

    async def add_lora(self, lora_name: str, lora_alias: str, lora_path: str):
        logging.info(f"Adding LoRA {lora_name} from {lora_path}")
        lora_id = self.lora_id_counter.inc(1)
        lora_request = create_lora_request(lora_name, lora_id, lora_path, None)
        self.lora_requests[lora_request.lora_name] = lora_request
        result = await self.vllm_engine.add_lora(lora_request)
        logging.info(f"LoRA {lora_name} added to engine: {result}")
        return result

    async def remove_lora(self, lora_name: str, lora_alias: str):
        logging.info(f"Removing LoRA {lora_name}")
        if lora_name not in self.lora_requests:
            raise ValueError(f"LoRA adapter {lora_name} not found in registry")
        lora_request = get_lora_request(lora_name, self.lora_requests)
        result = await self.vllm_engine.remove_lora(lora_request.lora_int_id)
        del self.lora_requests[lora_name]
        return result

    async def pin_lora(self, lora_name: str, lora_alias: str):
        lora_request = get_lora_request(lora_name, self.lora_requests)
        loaded = await self.vllm_engine.add_lora(lora_request)
        return loaded and await self.vllm_engine.pin_lora(
            lora_request.lora_int_id)

    def _extract_lora_adapter(self, raw_request, decoded_payload):
        """
        Get lora adapter name from request headers or payload.
        """
        adapter_name = None

        if SAGEMAKER_ADAPTER_IDENTIFIER_HEADER in raw_request.get_properties():
            adapter_name = raw_request.get_property(
                SAGEMAKER_ADAPTER_IDENTIFIER_HEADER)
            logging.debug(f"Found adapter in headers: {adapter_name}")
        elif "adapter" in decoded_payload:
            adapter_name = decoded_payload.pop("adapter")
            logging.debug(f"Found adapter in payload: {adapter_name}")

        return adapter_name


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


async def register_adapter(inputs: Input):
    """
    Registers lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name
    adapter_path = inputs.get_property("src")
    adapter_preload = inputs.get_as_string("preload").lower(
    ) == "true" if inputs.contains_key("preload") else True
    adapter_pin = inputs.get_as_string(
        "pin").lower() == "true" if inputs.contains_key("pin") else False

    outputs = Output()
    loaded = False
    try:
        if not os.path.exists(adapter_path):
            raise ValueError(
                f"Only local LoRA models are supported. {adapter_path} is not a valid path"
            )

        if not adapter_preload and adapter_pin:
            raise ValueError("Can not set preload to false and pin to true")

        if adapter_preload:
            loaded = await service.add_lora(adapter_name, adapter_alias,
                                            adapter_path)

        if adapter_pin:
            await service.pin_lora(adapter_name, adapter_alias)
        service.adapter_registry[adapter_name] = inputs
    except Exception as e:
        logging.debug(f"Failed to register adapter: {e}", exc_info=True)
        if loaded:
            logging.info(
                f"LoRA adapter {adapter_alias} was successfully loaded, but failed to pin, unloading ..."
            )
            await service.remove_lora(adapter_name, adapter_alias)
        if any(msg in str(e)
               for msg in ("No free lora slots",
                           "greater than the number of GPU LoRA slots")):
            raise MemoryError(str(e))
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(
        f"Registered adapter {adapter_alias} from {adapter_path} successfully")
    result = {"data": f"Adapter {adapter_alias} registered"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs


async def update_adapter(inputs: Input):
    """
    Updates lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name
    adapter_path = inputs.get_property("src")
    adapter_preload = inputs.get_as_string("preload").lower(
    ) == "true" if inputs.contains_key("preload") else True
    adapter_pin = inputs.get_as_string(
        "pin").lower() == "true" if inputs.contains_key("pin") else False

    if adapter_name not in service.adapter_registry:
        raise ValueError(f"Adapter {adapter_alias} not registered.")

    outputs = Output()
    try:
        if not adapter_preload and adapter_pin:
            raise ValueError("Can not set load to false and pin to true")

        old_adapter = service.adapter_registry[adapter_name]
        old_adapter_path = old_adapter.get_property("src")
        if adapter_path != old_adapter_path:
            raise NotImplementedError(
                f"Updating adapter path is not supported.")

        old_adapter_preload = old_adapter.get_as_string("preload").lower(
        ) == "true" if old_adapter.contains_key("preload") else True
        if adapter_preload != old_adapter_preload:
            if adapter_preload:
                await service.add_lora(adapter_name, adapter_alias,
                                       adapter_path)
            else:
                await service.remove_lora(adapter_name, adapter_alias)

        old_adapter_pin = old_adapter.get_as_string("pin").lower(
        ) == "true" if old_adapter.contains_key("pin") else False
        if adapter_pin != old_adapter_pin:
            if adapter_pin:
                await service.pin_lora(adapter_name, adapter_alias)
            else:
                raise NotImplementedError(f"Unpin adapter is not supported.")
        service.adapter_registry[adapter_name] = inputs
    except Exception as e:
        logging.debug(f"Failed to update adapter: {e}", exc_info=True)
        if any(msg in str(e)
               for msg in ("No free lora slots",
                           "greater than the number of GPU LoRA slots")):
            raise MemoryError(str(e))
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(f"Updated adapter {adapter_alias} successfully")
    result = {"data": f"Adapter {adapter_alias} updated"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs


async def unregister_adapter(inputs: Input):
    """
    Unregisters lora adapter from the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name

    if adapter_name not in service.adapter_registry:
        raise ValueError(f"Adapter {adapter_alias} not registered.")

    outputs = Output()
    try:
        await service.remove_lora(adapter_name, adapter_alias)
        del service.adapter_registry[adapter_name]
    except Exception as e:
        logging.debug(f"Failed to unregister adapter: {e}", exc_info=True)
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(f"Unregistered adapter {adapter_alias} successfully")
    result = {"data": f"Adapter {adapter_alias} unregistered"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs
