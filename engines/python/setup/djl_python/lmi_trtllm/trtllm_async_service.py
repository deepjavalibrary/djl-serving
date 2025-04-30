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

# Heavily inspired by https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/openai_server.py
import asyncio
import logging
import signal
import types
from http import HTTPStatus
from typing import AsyncGenerator, AsyncIterator, List, Tuple, TypedDict, Union, Optional

from openai.types.chat import ChatCompletionMessageParam

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                CompletionResponseChoice,
                                                ErrorResponse, ModelCard,
                                                ModelList, UsageInfo,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs, CompletionPostprocArgs, chat_response_post_processor,
    chat_stream_post_processor, completion_response_post_processor,
    completion_stream_post_processor)

from djl_python.async_utils import create_non_stream_output, handle_streaming_response, ProcessedRequest
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.encode_decode import decode

from .request_response_utils import trtllm_non_stream_output_formatter, trtllm_stream_output_formatter, \
    convert_lmi_schema_to_completion_request, lmi_with_details_stream_output_formatter, lmi_stream_output_formatter, lmi_with_details_non_stream_output_formatter, lmi_non_stream_output_formatter

logger = logging.getLogger(__name__)


class ConversationMessage(TypedDict):
    role: str
    content: str


def parse_chat_message_content(
        message: ChatCompletionMessageParam, ) -> List[ConversationMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    # for Iterable[ChatCompletionContentPartTextParam]
    texts: List[str] = []
    for part in content:
        part_type = part["type"]
        if part_type == "text":
            text = part["text"]
            texts.append(text)
        else:
            raise NotImplementedError(f"{part_type} is not supported")

    text_prompt = "\n".join(texts)
    return [ConversationMessage(role=role, content=text_prompt)]


class TensorRTLlmAsyncService:

    def __init__(self):
        self.trt_configs = None
        self.llm = None
        self.tokenizer = None
        self.model_name = None
        self.postproc_worker_enabled = False
        self.initialized = False

    def initialize(self, properties: dict):
        self.trt_configs = TensorRtLlmProperties(**properties)
        # In this handler, we expect the front-end to pre-compile the model into a TRTLLM engine.
        # That means we do not expose build config customization here, and only set runtime configs
        self.llm = LLM(
            self.trt_configs.model_id_or_path,
            **self.trt_configs.get_llm_kwargs(),
        )
        self.tokenizer = self.llm.tokenizer
        self.model_name = self.trt_configs.model_id_or_path
        self.postproc_worker_enabled = self.llm.args.num_postprocess_workers > 0
        self.initialized = True


    def preprocess_requests(self, inputs: Input) -> ProcessedRequest:
        batch = inputs.get_batches()
        assert len(batch) == 1, "only one request per batch allowed"
        raw_request = batch[0]
        content_type = raw_request.get_property("Content-Type")
        decoded_payload = decode(raw_request, content_type)
        accumulate_chunks = False
        include_prompt = False
        if "model" not in decoded_payload:
            decoded_payload["model"] = self.model_name
        if "prompt" in decoded_payload:
            request = CompletionRequest(**decoded_payload)
            invoke_call = self.openai_completion
            non_stream_output_formatter = trtllm_non_stream_output_formatter
            stream_output_formatter = trtllm_stream_output_formatter
        elif "messages" in decoded_payload:
            request = ChatCompletionRequest(**decoded_payload)
            invoke_call = self.openai_chat
            non_stream_output_formatter = trtllm_non_stream_output_formatter
            stream_output_formatter = trtllm_stream_output_formatter
        elif "inputs" in decoded_payload:
            request, include_details, include_prompt = convert_lmi_schema_to_completion_request(decoded_payload)
            invoke_call = self.openai_completion
            non_stream_output_formatter = lmi_with_details_non_stream_output_formatter if include_details else lmi_non_stream_output_formatter
            stream_output_formatter = lmi_with_details_stream_output_formatter if include_details else lmi_stream_output_formatter
            accumulate_chunks = True
        else:
            raise RuntimeError("invalid payload. must contain prompt, inputs, or messages")
        processed_request = ProcessedRequest(
            request,
            invoke_call,
            non_stream_output_formatter,
            stream_output_formatter,
            accumulate_chunks,
            include_prompt,
        )
        return processed_request

    async def inference(self, inputs: Input) -> Union[Output, AsyncGenerator[Output, None]]:
        try:
            processed_request = self.preprocess_requests(inputs)
        except Exception as e:
            logger.exception("Input parsing failed")
            output = create_non_stream_output("", error=f"Input parsing failed: {e}", code=424)
            return output

        response = await processed_request.inference_invoker(processed_request.request)
        if isinstance(response, types.AsyncGeneratorType):
            return handle_streaming_response(
                response,
                processed_request.stream_output_formatter,
                accumulate_chunks=processed_request.accumulate_chunks,
                include_prompt=processed_request.include_prompt,
                tokenizer=self.tokenizer,
                request=processed_request.request,
            )

        return processed_request.non_stream_output_formatter(
            response,
            request=processed_request.request,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=status_code.value)
        return error_response

    async def openai_chat(self, request: ChatCompletionRequest) -> Union[ErrorResponse, ChatCompletionResponse, AsyncGenerator[str, None]]:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
            async for res in promise:
                pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                for pp_res in pp_results:
                    yield pp_res
            yield f"data: [DONE]\n\n"

        async def create_chat_response(
                promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                return promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                return post_processor(promise, args)

        try:
            conversation: List[ConversationMessage] = []
            for msg in request.messages:
                conversation.extend(parse_chat_message_content(msg))
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            prompt: str = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)
            postproc_args = ChatPostprocArgs.from_request(request)
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
                disaggregated_params=disaggregated_params
            )
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return response_generator
            else:
                response = await create_chat_response(promise, postproc_params)
                return response
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest) -> Union[ErrorResponse, CompletionResponse, AsyncGenerator[str, None]]:

        def merge_promises(
                promises: List[RequestOutput],
                postproc_params_collections: List[Optional[PostprocParams]]
        ) -> AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput, postproc_params: Optional[PostprocParams]):
                async for output in promise:
                    await outputs.put((output, postproc_params))
                finished[i] = True

            _tasks = [
                asyncio.create_task(producer(i, promise, postproc_params))
                for i, (promise, postproc_params) in enumerate(zip(promises, postproc_params_collections))
            ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        async def create_completion_generator(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]):
            async for request_output, postproc_params in generator:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result
                for pp_res in pp_result:
                    yield pp_res
            yield f"data: [DONE]\n\n"

        async def create_completion_response(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]) -> CompletionResponse:
            all_choices: List[CompletionResponseChoice] = []
            num_prompt_tokens = num_gen_tokens = 0
            async for request_output, postproc_params in generator:
                pp_result: CompletionResponse
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result

                choices, usage = pp_result.choices, pp_result.usage
                all_choices.extend(choices)
                num_prompt_tokens += usage.prompt_tokens
                num_gen_tokens += usage.completion_tokens

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model_name,
                choices=all_choices,
                usage=usage_info,
            )
            return response

        try:
            if isinstance(request.prompt, str) or \
                    (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            sampling_params = request.to_sampling_params()
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            generator = merge_promises(promises, postproc_params_collection)
            if request.stream:
                response_generator = create_completion_generator(
                    generator)
                return response_generator
            else:
                response = await create_completion_response(
                    generator)
                return response
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.exception(f"Encountered an exception: {str(e)}")
            return self.create_error_response(str(e))


service = TensorRTLlmAsyncService()


async def handle(inputs: Input) -> Optional[Output]:
    if not service.initialized:
        service.initialize(inputs.get_properties())
        logger.info("trtllm service initialized")
    if inputs.is_empty():
        return None

    outputs = await service.inference(inputs)
    return outputs
