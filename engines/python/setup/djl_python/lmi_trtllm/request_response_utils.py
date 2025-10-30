import json
from typing import Callable, Union, Tuple, List
from tensorrt_llm.serve.openai_protocol import (
    ErrorResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    CompletionRequest,
    CompletionLogProbs,
)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from djl_python.async_utils import create_non_stream_output
from djl_python.outputs import Output


class ProcessedRequest:

    def __init__(
        self,
        trtllm_request: Union[CompletionRequest, ChatCompletionRequest],
        inference_invoker: Callable,
        non_stream_output_formatter: Callable,
        stream_output_formatter: Callable,
        accumulate_chunks: bool,
        include_prompt: bool,
    ):
        self.trtllm_request = trtllm_request
        self.inference_invoker = inference_invoker
        # We need access to both the stream and non-stream output formatters here
        # because even with streaming requests, there may be some errors before inference that
        # result in a return of ErrorResponse object instead of AsyncGenerator
        self.non_stream_output_formatter = non_stream_output_formatter
        self.stream_output_formatter = stream_output_formatter
        self.accumulate_chunks = accumulate_chunks
        self.include_prompt = include_prompt
        self.lora_request = None


def convert_lmi_schema_to_completion_request(
    payload: dict, ) -> Tuple[CompletionRequest, bool, bool]:
    parameters = payload.get("parameters", {})

    completion_dict = {
        "prompt": payload.pop("inputs"),
        "model": payload.pop("model"),
        "max_tokens": parameters.pop("max_new_tokens", 30),
        "echo": parameters.pop("return_full_text", False),
        "truncate_prompt_tokens": parameters.pop("truncate", None),
        "n": parameters.pop("top_n_tokens", 1),
        "ignore_eos": parameters.pop("ignore_eos_token", False),
        "stream": payload.pop("stream", False),
    }
    # TRTLLM does not support logprobs in completions API. If provided, rely on TRTLLM validation error
    include_details_in_response = False
    include_prompt = False
    if completion_dict["stream"]:
        completion_dict["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True
        }
        include_prompt = completion_dict.pop("echo", False)
    if parameters.pop("details", False):
        include_details_in_response = True
        if parameters.pop("decoder_input_details", False):
            completion_dict["return_context_logits"] = 1
    do_sample = parameters.pop("do_sample", None)
    # when do_sample is None, just passthrough sampling params as sampling is dictated by the value of other params
    # when do_sample is False, set sampling params such that we disable sampling
    if do_sample is not None and not do_sample:
        parameters["temperature"] = 0.0

    completion_dict.update(parameters)

    return CompletionRequest(
        **completion_dict), include_details_in_response, include_prompt


def convert_completion_response_to_lmi_schema(
        response: CompletionResponse,
        request: CompletionRequest = None,
        include_details: bool = False,
        tokenizer: TokenizerBase = None) -> Output:
    primary_choice = response.choices[0]
    lmi_response = {"generated_text": primary_choice.text}
    if not include_details:
        return create_non_stream_output(lmi_response)
    details = {
        "finish_reason": primary_choice.stop_reason,
        "generated_tokens": response.usage.completion_tokens,
        "seed": request.seed,
    }
    lmi_response["details"] = details
    output = create_non_stream_output(lmi_response)
    return output


def convert_completion_chunk_response_to_lmi_schema(
    chunk: str,
    include_details: bool = False,
    history: List[str] = None,
    request: CompletionRequest = None,
    include_prompt: bool = False,
    tokenizer: TokenizerBase = None,
    **_,
) -> Tuple[str, bool, List[str]]:
    # TRTLLM returns chunks in string format, and the conversion process to TGI
    # currently converts the string to an object, and then the object back to a string.
    # It's much easier to work with the object instead of manipulating the string, but inefficient
    trimmed_chunk = chunk[6:].strip()
    if trimmed_chunk == '[DONE]':
        data = ""
        return data, True, history

    trt_completion_chunk = json.loads(trimmed_chunk)
    if "error" in trt_completion_chunk:
        return json.dumps(trt_completion_chunk,
                          ensure_ascii=False), True, history

    if len(trt_completion_chunk["choices"]) == 0:
        # penultimate chunk
        return "", False, history
    choice = trt_completion_chunk["choices"][0]
    index = choice["index"]
    token_text = choice["text"]
    history.append(token_text)
    finish_reason = choice["finish_reason"]
    stop_reason = choice["stop_reason"]
    usage = trt_completion_chunk["usage"]

    # TODO: TokenId and LogProb here
    token = {
        "id": None,
        "text": token_text,
        "logprob": None,
    }
    tgi_chunk = {
        "index": index,
        "token": token,
        "generated_text": None,
        "details": None,
    }
    generation_finished = finish_reason is not None or stop_reason is not None
    if generation_finished:
        generated_text = ''.join(history)
        if include_prompt:
            generated_text = request.prompt + generated_text
        tgi_chunk["generated_text"] = generated_text
        if include_details:
            details = {
                "finish_reason": finish_reason or stop_reason,
                "seed": request.seed,
                "generated_tokens": usage["completion_tokens"] + 1,
                "input_length": usage["prompt_tokens"],
            }
            tgi_chunk["details"] = details
    json_str = json.dumps(tgi_chunk, ensure_ascii=False)
    return json_str, False, history


def lmi_with_details_non_stream_output_formatter(
    response: CompletionResponse,
    request: CompletionRequest = None,
    tokenizer: TokenizerBase = None,
) -> Output:
    return convert_completion_response_to_lmi_schema(response,
                                                     include_details=True,
                                                     request=request,
                                                     tokenizer=tokenizer)


def lmi_non_stream_output_formatter(
    response: CompletionResponse,
    request: CompletionRequest = None,
    tokenizer: TokenizerBase = None,
) -> Output:
    return convert_completion_response_to_lmi_schema(response,
                                                     include_details=False,
                                                     request=request,
                                                     tokenizer=tokenizer)


def lmi_with_details_stream_output_formatter(
    chunk: str,
    **kwargs,
) -> Tuple[str, bool, List[str]]:
    return convert_completion_chunk_response_to_lmi_schema(
        chunk, include_details=True, **kwargs)


def lmi_stream_output_formatter(
    chunk: str,
    **kwargs,
) -> Tuple[str, bool, List[str]]:
    return convert_completion_chunk_response_to_lmi_schema(chunk, **kwargs)


def trtllm_non_stream_output_formatter(
    response: Union[ErrorResponse, ChatCompletionResponse, CompletionResponse],
    **_,
) -> Output:
    if isinstance(response, ErrorResponse):
        return create_non_stream_output("",
                                        error=response.message,
                                        code=response.code)
    response_data = response.model_dump_json()
    return create_non_stream_output(response_data)


def trtllm_stream_output_formatter(
    chunk: str,
    **_,
) -> Tuple[str, bool]:
    # trtllm returns responses in sse format, 'data: {...}'
    trimmed_chunk = chunk[6:].strip()
    if trimmed_chunk == '[DONE]':
        data = ""
        last = True
    else:
        data = trimmed_chunk
        last = False
    return data, last
