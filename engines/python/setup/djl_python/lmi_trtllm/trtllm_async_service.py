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
from typing import AsyncGenerator, AsyncIterator, List, Tuple, TypedDict, Union, Callable, Optional

from openai.types.chat import ChatCompletionMessageParam

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import LLM, BuildConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    ChatMessage, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, CompletionResponseStreamChoice,
    CompletionStreamResponse, DeltaMessage, ErrorResponse, FunctionCall,
    ModelCard, ModelList, ToolCall, UsageInfo)

from djl_python.async_utils import create_non_stream_output, handle_streaming_response, ProcessedRequest
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.encode_decode import decode

from .request_response_utils import trtllm_non_stream_output_formatter, trtllm_stream_output_formatter

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
        self.hf_configs = None
        self.llm = None
        self.tokenizer = None
        self.model_name = None
        self.initialized = False

    def initialize(self, properties: dict):
        self.hf_configs = TensorRtLlmProperties(**properties)
        # TODO: Support all the LLM args here, and in AOT for compilation
        self.llm = LLM(
            self.hf_configs.model_id_or_path,
            tokenizer=self.hf_configs.model_id_or_path,
            tensor_parallel_size=self.hf_configs.tensor_parallel_degree,
            trust_remote_code=self.hf_configs.trust_remote_code,
            build_config=BuildConfig(
                max_batch_size=self.hf_configs.max_rolling_batch_size,
                output_timing_cache="/tmp/model.cache",
            ),
            **self.hf_configs.get_extra_kwargs(),
        )
        self.model_name = self.hf_configs.model_id_or_path
        self.initialized = True

    def preprocess_requests(self, inputs: Input) -> ProcessedRequest:
        batch = inputs.get_batches()
        assert len(batch) == 1, "only one request per batch allowed"
        raw_request = batch[0]
        content_type = raw_request.get_property("Content-Type")
        decoded_payload = decode(raw_request, content_type)
        if "model" not in decoded_payload:
            decoded_payload["model"] = self.model_name
        if "prompt" in decoded_payload:
            request = CompletionRequest(**decoded_payload)
            invoke_call = self.openai_completion
        elif "messages" in decoded_payload:
            request = ChatCompletionRequest(**decoded_payload)
            invoke_call = self.openai_chat
        else:
            raise RuntimeError("invalid payload. must contain prompt, inputs, or messages")
        processed_request = ProcessedRequest(
            request,
            invoke_call,
            trtllm_non_stream_output_formatter,
            trtllm_stream_output_formatter,
            False,
            False,
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
                request=processed_request.request,
            )

        return processed_request.non_stream_output_formatter(
            response,
            request=processed_request.request,
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

        def stream_usage_info(prompt_tokens: int, completion_tokens: int):
            if request.stream_options and request.stream_options.include_usage and \
                    request.stream_options.continuous_usage_stats:
                usage = UsageInfo(prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens,
                                  total_tokens=prompt_tokens +
                                               completion_tokens)
            else:
                usage = None
            return usage

        def create_logprobs(token_ids: List[int],
                            logprobs: List[float]) -> ChatCompletionLogProbs:
            assert len(token_ids) == len(logprobs), \
                "token_ids and logprobs have different lengths"
            content: List[ChatCompletionLogProbsContent] = []
            for token_id, logprob in zip(token_ids, logprobs):
                token = self.tokenizer.decode(token_id)
                # returning multiple logprobs is not supported
                first_logprob = ChatCompletionLogProbsContent(
                    token=token, logprob=max(logprob, -9999.0),
                    bytes=list(token.encode("utf-8", errors="replace"))
                )
                content.append(first_logprob)
            chat_logprobs = ChatCompletionLogProbs(content=content)
            return chat_logprobs

        async def chat_stream_generator(promise: RequestOutput) -> AsyncGenerator[str, None]:
            first_iteration = True
            num_choices = 1 if request.n is None else request.n
            finish_reason_sent = [False] * num_choices
            role = get_role()

            def yield_first_chat(num_tokens: int, role: str = None, content: str = None):
                for i in range(num_choices):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(
                            role=role, content=content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        choices=[choice_data], model=self.model_name)
                    chunk.usage = stream_usage_info(num_tokens, 0)

                    data = chunk.model_dump_json(exclude_unset=True)
                    return data

            async for res in promise:
                prompt_tokens = len(res.prompt_token_ids)
                if first_iteration:
                    yield f"data: {yield_first_chat(prompt_tokens, role=role)} \n\n"

                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                            "role") == role:
                            last_msg_content = conversation[-1][
                                "content"]

                        if last_msg_content:
                            yield f"data: {yield_first_chat(prompt_tokens, content=last_msg_content)}\n\n"
                first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_text = output.text_diff
                    if request.tool_choice and type(
                            request.tool_choice
                    ) is ChatCompletionNamedToolChoiceParam:
                        delta_message = DeltaMessage(tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text))
                        ])
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    choice = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        finish_reason=None)
                    if request.logprobs:
                        logprobs = output.logprobs_diff
                        token_ids = output.token_ids_diff
                        choice.logprobs = create_logprobs(token_ids, logprobs)
                    if output.finish_reason is not None:
                        choice.finish_reason = output.finish_reason
                        choice.stop_reason = output.stop_reason
                        finish_reason_sent[i] = True
                    chunk = ChatCompletionStreamResponse(
                        choices=[choice], model=self.model_name)
                    chunk.usage = stream_usage_info(
                        prompt_tokens, output.length)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            if (request.stream_options
                    and request.stream_options.include_usage):
                completion_tokens = sum(output.length
                                        for output in promise.outputs)
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    choices=[], model=self.model_name, usage=final_usage)
                final_usage_data = final_usage_chunk.model_dump_json()
                yield f"data: {final_usage_data}\n\n"
            yield f"data: [DONE]\n\n"

        async def create_chat_response(promise: RequestOutput) -> ChatCompletionResponse:
            await promise.aresult()
            choices: List[ChatCompletionResponseChoice] = []
            role = get_role()
            for output in promise.outputs:
                if request.tool_choice and isinstance(
                        request.tool_choice,
                        ChatCompletionNamedToolChoiceParam):
                    message = ChatMessage(
                        role=role,
                        content="",
                        tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=output.text))
                        ])
                else:
                    message = ChatMessage(role=role, content=output.text)
                choice = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                )

                if request.logprobs:
                    choice.logprobs = create_logprobs(output.token_ids, output.logprobs)
                choices.append(choice)

            if request.echo:
                last_msg_content = ""
                if conversation and conversation[-1].get(
                        "content") and conversation[-1].get("role") == role:
                    last_msg_content = conversation[-1]["content"]
                for choice in choices:
                    full_message = last_msg_content + choice.message.content
                    choice.message.content = full_message

            num_prompt_tokens = len(promise.prompt_token_ids)
            num_generated_tokens = sum(
                len(output.token_ids) for output in promise.outputs)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = ChatCompletionResponse(
                model=self.model_name,
                choices=choices,
                usage=usage,
            )
            return response

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

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                streaming=request.stream,
            )
            if request.stream:
                response_generator = chat_stream_generator(promise)
                return response_generator
            else:
                response = await create_chat_response(promise)
                return response
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest) -> Union[ErrorResponse, CompletionResponse, AsyncGenerator[str, None]]:

        def merge_promises(promises: List[RequestOutput]) -> AsyncIterator[Tuple[int, RequestOutput]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput):
                async for output in promise:
                    await outputs.put((i, output))
                finished[i] = True

            _tasks = [asyncio.create_task(producer(i, promise))
                      for i, promise in enumerate(promises)
                      ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        async def create_completion_generator(generator: AsyncIterator[Tuple[int, RequestOutput]],
                                              num_choices: int):
            num_repsonse_per_request = 1 if request.n is None else request.n
            echoed = [False] * num_choices
            async for prompt_idx, requst_output in generator:
                prompt = requst_output.prompt
                for gen_idx, output in enumerate(requst_output.outputs):
                    response_idx = prompt_idx * num_repsonse_per_request + gen_idx
                    delta_text = output.text_diff
                    if request.echo and not echoed[response_idx]:
                        delta_text = prompt + delta_text
                        echoed[response_idx] = True
                    response = CompletionStreamResponse(
                        model=self.model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=response_idx,
                                text=delta_text,
                                stop_reason=output.stop_reason,
                                finish_reason=output.finish_reason,
                            )
                        ])
                    response_json = response.model_dump_json(
                        exclude_unset=False)
                    yield f"data: {response_json}\n\n"
            yield f"data: [DONE]\n\n"

        async def create_completion_response(generator: AsyncIterator[Tuple[int, RequestOutput]],
                                             num_choices: int):
            choices = [None] * num_choices
            num_repsonse_per_request = 1 if request.n is None else request.n
            num_prompt_tokens = num_gen_tokens = 0
            async for prompt_idx, request_output in generator:
                num_prompt_tokens += len(request_output.prompt_token_ids)
                for gen_idx, output in enumerate(request_output.outputs):
                    num_gen_tokens += len(output.token_ids)
                    output_text = output.text
                    if request.echo:
                        output_text = request_output.prompt + output_text
                    idx = prompt_idx * num_repsonse_per_request + gen_idx
                    choice = CompletionResponseChoice(
                        index=idx,
                        text=output_text,
                        stop_reason=output.stop_reason,
                        finish_reason=output.finish_reason,
                    )
                    choices[idx] = choice

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model_name,
                choices=choices,
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
            sampling_params = request.to_sampling_params()
            for prompt in prompts:
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    streaming=request.stream,
                )
                promises.append(promise)
            generator = merge_promises(promises)
            num_choices = len(prompts) if request.n is None else len(prompts) * request.n
            if request.stream:
                response_generator = create_completion_generator(generator, num_choices)
                return response_generator
            else:
                response = await create_completion_response(generator, num_choices)
                return response
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
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
