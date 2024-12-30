# LMI handlers Inference API Schema

This document provides the default API schema for the inference endpoints (`/invocations`, `/predictions/<model_name>`) when using the built-in inference handlers in LMI containers.
This schema is applicable to our latest release, v0.31.0.
Documentation for previous releases is available on our GitHub on the relevant version branch (e.g. 0.27.0-dlc).

LMI provides two distinct schemas depending on what type of batching you use:

* Rolling Batch/Continuous Batch Schema
* Dynamic Batch/Static Batch Schema

Both batching mechanisms support streaming, and the response format for streaming differs from the response format for non-streaming.

## Rolling Batch/Continuous Batch Schema

### Request Schema

Request Body Fields:

| Field Name   | Field Type                                    | Required | Possible Values                                                                                                                                     |
|--------------|-----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `inputs`     | string                                        | yes      | example: "What is Deep Learning"                                                                                                                    |
| `parameters` | [GenerationParameters](#generationparameters) | no       | See the [GenerationParameters](#generationparameters) documentation. This is a dictionary containing parameters that control the decoding behavior. |
| `stream`     | boolean                                       | no       | `true`, `false` (default)                                                                                                                           |

Example request using curl
```
curl -X POST https://my.sample.endpoint.com/invocations \
  - H 'Content-Type: application/json' \
  - d '
    {
        "inputs" : "What is Deep Learning?", 
        "parameters" : {
            "do_sample": true,
            "max_new_tokens": 256,
            "details": true,
        },
        "stream": true, 
    }'
```

### Response Schema

When not using streaming (the default), the response is returned as application/json content-type:

| Field Name       | Field Type          | Always Returned                           | Possible Values                                                                     | 
|------------------|---------------------|-------------------------------------------|-------------------------------------------------------------------------------------|
| `generated_text` | string              | yes                                       | The result of the model generation. Example: "Deep Learning is a really cool field" |
| `details`        | [Details](#details) | no, only when `parameters.details = true` | See the [Details](#details) documentation                                           |

Example response:
```
{
    "generated_text": "Deep Learning is a really cool field",
    "details": {
        "finish reason": "length",
        "generated_tokens": 8,
        "inputs": "What is Deep Learning?",
        "tokens": [<Token1>, <Token2>, ...]
    }
}
```

When using streaming (i.e. `"stream": true`, or using `option.output_formatter=jsonlines`), the response is returned as application/jsonlines content-type.
The response is returned token by token, and each "line" in the response is a [Token](#token) object.
The final "line" in the response will also contain the additional fields `generated_text`, and `details`.

| Field Name       | Field Type          | Always Returned                       | Possible Values                           |
|------------------|---------------------|---------------------------------------|-------------------------------------------|
| `token`          | [Token](#token)     | yes                                   | See the [Token](#token) documentation     |
| `generated_text` | string              | no, only on the last generated "line" | "Deep Learning is a really cool field"    |
| `details`        | [Details](#details) | no, only on the last generated "line" | See the [Details](#details) documentation |

Example response:
```
{"token": {"id": 304, "text": "Deep ", "log_prob": -0.052432529628276825}}
{"token": {"id": 11157, "text": " Learning", "log_prob": -1.2865009307861328}}
{"token": {"id": 278, "text": " is", "log_prob": -0.007458459585905075}}
... more tokens until the last one
{
    "token": {"id": 5972, "text": " field.", "log_prob": -0.6950479745864868}, 
    "generated_text": "Deep Learning is a really cool field.", 
    "details": {"finish_reason": "length", "generated_tokens": 100, "inputs": "What is Deep Learning?"}
}
```

When using streaming, if you want Server Side Events, then you could use `option.output_formatter=sse`. 
If you `stream=True`, the default `output_formatter` is `jsonlines`. 
So you would want to explicitly provide `option.output_formatter=sse` when you want SSE with streaming. 
Check out `TGI_COMPAT` option below, enabling that option will make SSE as the default formatter with streaming. 
When using SSE the jsonline will have the prefix `data`. 

Example response:
```
data:{"token": {"id": 304, "text": "Deep ", "log_prob": -0.052432529628276825}}
data:{"token": {"id": 11157, "text": " Learning", "log_prob": -1.2865009307861328}}
data:{"token": {"id": 278, "text": " is", "log_prob": -0.007458459585905075}}
... more tokens until the last one
data:{
    "token": {"id": 5972, "text": " field.", "log_prob": -0.6950479745864868}, 
    "generated_text": "Deep Learning is a really cool field.", 
    "details": {"finish_reason": "length", "generated_tokens": 100, "inputs": "What is Deep Learning?"}
}
```


#### Error Responses

Errors can typically happen in two places:

- Before inference has started
- During token generation (in the middle of inference)

For both streaming and non-streaming responses, errors that occur before inference has started will return the appropriate 4XX or 5XX HTTP status code.

If an error occurs during inference, then the behavior differs between streaming and non-streaming use-cases:

- streaming responses will return an HTTP status code of 200 upon the first token generated and returned to the client
- non-streaming responses will return the appropriate 4XX or 5XX HTTP status code

The response payload for errors is described for various scenarios.

If there is an issue with payload validation, then the response details are:

```
{
    "error": "<error details>",
    "code": 424
}
```

If there is an issue with the request (such as bad generation parameter value), or an error occurs during generation, the response content will be:

When not using streaming:

```
{
    "generated_text": "", 
    "details": {
        "finish_reason": "error", 
        "generated_tokens": null, 
        "inputs": null, 
        "tokens": null
    }
}
```

When using streaming:

```
{
    "token": {"id": -1, "text": "", "log_prob": -1, "special_token": true}, 
    "generated_text": "", 
    "details": {"finish_reason": "error", "generated_tokens": null, "inputs": null}}
}
```

## Response with TGI compatibility

In order to get the same response output as HuggingFace's Text Generation Inference, you can use the env `OPTION_TGI_COMPAT=true` or `option.tgi_compat=true` in your serving.properties. 
Right now, DJLServing for LMI with rolling batch has minor differences in the response schema compared to TGI. 

This feature is designed for customers transitioning from TGI, making their lives easier by allowing them to continue using their client-side code without any special modifications for our LMI containers or DJLServing.
Enabling the tgi_compat option would make the response look like below:

When not using streaming: Response will be a JSONArray, instead of JSONObject. 
```
[
    {
        "generated_text": "Deep Learning is a really cool field",
        "details": {
            "finish reason": "length",
            "generated_tokens": 8,
            "inputs": "What is Deep Learning?",
            "tokens": [<Token1>, <Token2>, ...]
        }
    }
]
```

When using streaming: Response will be Server Sent Events (text/event-stream) which will prefix with `data:`

```
data: {
    "token": {"id": 5972, "text": " field.", "log_prob": -0.6950479745864868}, 
    "generated_text": "Deep Learning is a really cool field.", 
    "details": {"finish_reason": "length", "generated_tokens": 100, "inputs": "What is Deep Learning?"}
}
```


## Dynamic Batch/Static Batch Schema

### Request Schema

Request Body Fields:

| Field Name   | Field Type                                    | Required | Possible Values                                                                                                                                     |
|--------------|-----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `inputs`     | string, array of strings                      | yes      | example: "What is Deep Learning", ["What is Deep Learning", "How many ways can I peel an orange"]                                                   |
| `parameters` | [GenerationParameters](#generationparameters) | no       | See the [GenerationParameters](#generationparameters) documentation. This is a dictionary containing parameters that control the decoding behavior. |

Example request using curl
```
curl -X POST https://my.sample.endpoint.com/invocations \
  - H 'Content-Type: application/json' \
  - d '
    {
        "inputs" : "What is Deep Learning?", 
        "parameters" : {
            "do_sample": true,
            "max_new_tokens": 256,
        }
    }'
```

### Response Schema

When not using streaming (this is the default), the response is returned as application/json content-type:

| Field Name       | Field Type          | Always Returned                           | Possible Values                                                                     | 
|------------------|---------------------|-------------------------------------------|-------------------------------------------------------------------------------------|
| `generated_text` | string              | yes                                       | The result of the model generation. Example: "Deep Learning is a really cool field" |

The response is a list of objects with one field each, `generated_text`. Each object in the output corresponds to the prompt with the same index in the input.

Example response:
```
[
  {
    "generated_text": "Deep Learning is a really cool field"
  }
]
```

When using streaming (i.e. using `option.enable_streaming=true`), the response is returned as application/jsonlines content-type.
The response is returned token by token. 

| Field Name | Field Type | Always Returned                       | Possible Values                        |
|------------|------------|---------------------------------------|----------------------------------------|
| `outputs`  | string     | no, only on the last generated "line" | "Deep Learning is a really cool field" |

Example response:
```
{"outputs": ["This"]}
{"outputs": ["opportunity"]}
{"outputs": ["is"]}
...more outputs until the last one
```

#### Error Responses

When using dynamic batching, errors are returned with HTTP response code 424 and content:

``` 
{
    "code": 424,
    "message": "invoke handler failure",
    "error": "<error details"
}
```


## API Object Schemas

The following sections describe each of the request or response objects in more detail.

### GenerationParameters

The following parameters are available with every rolling batch backend (vLLM, lmi-dist, tensorrt-llm, hf-accelerate)

```
"parameters": {
  'do_sample' : boolean (default = False),
  'seed' : integer (default = ramdom value),
  'temperature' : float (default= 1.0),
  'repetition_penalty': float (default= 1.0),
  'top_k' : integer (default = 0), 
  'top_p' : float (default= 1.0),
  'max_new_tokens' : integer (default = 30),
  'details' : boolean (default = false, details only available for rolling batch),
  'return_full_text': boolean (default = false),
  'stop_sequences' : list[str] (default = None)
  'decoder_input_details' : boolean (default = false, only for vllm, lmi-dist)
}
```

If you are not specifying a specific engine or rolling batch implementation, we recommend you stick to the parameters above.

If you are deploying with a specific backend, additional parameters are available that are unique to the specific backend.

#### Additional LMI Dist Generation parameters

```
LmiDistRollingBatchParameters : {
    'typical_p' : float (default= 1.0),
    'truncate' : integer (default = None),
    'ignore_eos_token' : boolean (default = false),
    'top_k' : integer (default = -1),
    'min_p': float (default = 0.0),
    'presence_penalty': float (default = 0.0),
    'frequency_penalty' : float (default = 0.0),
    'n': integer (default = 1), (set this greater than 1 to get mutliple sequences. only works with non-streaming case) 
    'best_of': integer (default = 1)
    'num_beams': integer (default = 1), (set this greater than 1 to enable beam search. only works with non-streaming case)
    'length_penalty' : float (default = 1.0),
    'early_stopping' : boolean (default = false),
    'stop_token_ids': list (default = None),
    'include_stop_str_in_output' : boolean (default = false),
    'ignore_eos_token' : boolean (default = false),
    'logprobs' : int (default = None),
    'prompt_logprobs' : int (default = None),
    'skip_special_tokens': boolean (default = true),
    'spaces_between_special_tokens': boolean (default = true),
}
```

Decoding methods supported in LmiDist : Greedy (Default) and Sampling.


#### Additional vLLM Generation Parameters

```
vLLMRollingBatchParameters : {
    'top_k' : integer (default = -1)
    'min_p': float (default = 0.0),
    'presence_penalty': float (default = 0.0),
    'frequency_penalty' : float (default = 0.0),
    'n': integer (default = 1), (set this greater than 1 to get mutliple sequences. only works with non-streaming case) 
    'best_of': integer (default = 1)
    'num_beams': integer (default = 1), (set this greater than 1 to enable beam search. only works with non-streaming case)
    'length_penalty' : float (default = 1.0),
    'early_stopping' : boolean (default = false),
    'stop_token_ids': list (default = None),
    'include_stop_str_in_output' : boolean (default = false),
    'ignore_eos_token' : boolean (default = false),
    'logprobs' : int (default = None),
    'prompt_logprobs' : int (default = None),
    'skip_special_tokens': boolean (default = true),
    'spaces_between_special_tokens': boolean (default = true),
}
```

Decoding methods supported in vLLM : Greedy (Default), Sampling, and Beam search.

#### Additional TensorRT-LLM Generation Parameters 

For TensorRTLLM handler, some of the common parameters have different default values. 

```
TensorRTLLMRollingBatchParameters : {
    # Common parameters with different default values
    'temperature' : float (default = 0.8 when greedy, 1.0 when do_sample=True),
    'top_k' : integer (default = 0 when greedy, 5 when do_sample=True), 
    'top_p' : float (default= 0 when greedy, 0.85 when do_sample=True),
    
    # parameters specific to TensorRT-LLM
    'min_length' : integer (default = 1)
    'bad_sequences' : list[str] (default = None), 
    'stop' : boolean, 
    'presence_penalty': float,
    'length_penalty' : float, 
    'frequency_penalty': float
}
```

Decoding method supported in TensorRT-LLM : Greedy (Default) and Sampling.

NOTE: TensorRT-LLM C++ runtime, does not have the option `do_sample`. If top_k and top_p are 0, TensorRT-LLM automatically recognizes it as greedy.
So in order to do sampling, we set top_k, top_p and temperature values to a certain values. You can change these parameters for your use-case and pass them at runtime.

For those without default values, they remain optional. If these parameters are not provided, they will be ignored.

### Token

The token object represents a single generated token.
It contains the following fields:

| Field Name | Type   | Description                                                        | Example |
|------------|--------|--------------------------------------------------------------------|---------|
| `id`       | number | The token id used for encoding/decoding text                       | 45      |
| `text`     | string | the text representation of the toke                                | " of"   |
| `log_prob` | number | the log probability of the token (closer to 0 means more probable) | -0.12   |

Example:

```
{
  "token": {
    "id": 763, 
    "text": " In", 
    "log_prob": -3.977081060409546
  }
}
```

### Details

Additional details relevant to the generation. This is only available when using continuous batching.
You must specify `details=true` in the input `parameters`.

| Field Name          | Type                                       | Description                                                                                             | Example                                                 |
|---------------------|--------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `finish_reason`     | string enum                                | the reason for concluding generation                                                                    | `length`, `eos_token`, `stop_sequence`                  |
| `generated_tokens`  | number                                     | the number of tokens generated                                                                          | 128                                                     |
| `inputs`            | string                                     | the input/prompt used to start generation                                                               | "Deep Learning is"                                      |
| `tokens`            | array of [Tokens](#token)                  | An array of token objects, one for each token generated. Only returned in non-streaming use-cases       | See the [Tokens](#token) documentation                  |
| `prefill`           | array of [Tokens](#token)                  | An array of token objects, one for each prompt token.                                                   | See the [Tokens](#token) documentation                  |
| `best_of_sequences` | array of [BestOfSequence](#bestofsequence) | An array of BestOfSequence objects, one for each best of sequence. When either best_of or num_beams > 1 | See the [BestOfSequence](#bestofsequence) documentation |


Example:
```
"details": {
   "finish_reason": "length",
   "generated_tokens": 128,
   "inputs": "Deep Learning is"
   "tokens": [<Token1>, <Token2>, ...]
   "prefill": [<PromptToken1>, <PromptToken2>, ...]
}
```

### BestOfSequence

Generated text and its details is the one with the highest log probability. 
Other sequences are returned as best_of_sequences. 
You can enable this with n > 1. 
It is also returned when beam search is enabled with the option num_beams > 1. 

Note that best_of_sequences will only work with non-streaming case.

| Field Name         | Type                      | Description                                                | Example                                |
|--------------------|---------------------------|------------------------------------------------------------|----------------------------------------|
| `finish_reason`    | string enum               | the reason for concluding generation                       | `length`, `eos_token`, `stop_sequence` |
| `generated_tokens` | number                    | the number of tokens generated                             | 128                                    |
| `tokens`           | array of [Tokens](#token) | An array of token objects, one for each token generated.   | See the [Tokens](#token) documentation |
| `generated_text`   | string                    | The result of this sequence generation                     | `Haus ist wunderbar`                   |


## Custom pre- and post-processing

If you wish to create your own pre-processing and post-processing for our handlers, check out these guides [Custom input format schema guide](input_formatter_schema.md) and [Custom output format schema guide](output_formatter_schema.md).

This is an experimental use-case. The API signature, as well as implementation, is subject to change at any time.