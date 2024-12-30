# Chat Completions API Schema

This document describes the API schema for the chat completions endpoints (`v1/chat/completions`) when using the built-in inference handlers in LMI containers.
This schema is applicable to our latest release, v0.30.0, and is compatible with [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).
Documentation for previous releases is available on our GitHub on the relevant version branch (e.g. 0.30.0-dlc).

On SageMaker, Chat Completions API schema is supported with the `/invocations` endpoint without additional configurations.
If the request contains the "messages" field, LMI will treat the request as a chat completions style request, and respond
back with the chat completions response style.

When using the Chat Completions Schema, you should make sure that the model you are serving has a chat template.
The chat template ensures that the payload is tokenized appropriately for your model.
See [the HuggingFace documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) for more information.

This processing happens per request, meaning that you can support our [standard schema](lmi_input_output_schema.md),
as well as chat completions schema in the same endpoint.

Note: This is an experimental feature. The complete spec has not been implemented.
In particular, function calling is not currently available.
We are targeting function calling support in our next release, 0.32.0, planned for January 2025.

## Request Schema

Request Body Fields:

| Field Name          | Field Type                   | Required | Possible Values                           |
|---------------------|------------------------------|----------|-------------------------------------------|
| `messages`          | array of [Message](#message) | yes      | See the [Message](#message) documentation |
| `model`             | string                       | no       | example: "gpt2"                           |
| `frequency_penalty` | float                        | no       | Float number between -2.0 and 2.0.        |
| `logit_bias`        | dict                         | no       | example: {"2435": -100.0, "640": -100.0}  |
| `logprobs`          | boolean                      | no       | `true`, `false` (default)                 |
| `top_logprobs`      | int                          | no       | Integer between 0 and 20.                 |
| `max_tokens`        | int                          | no       | Positive integer.                         |
| `n`                 | int                          | no       | Positive integer.                         |
| `presence_penalty`  | float                        | no       | Float number between -2.0 and 2.0.        |
| `seed`              | int                          | no       | Integer.                                  |
| `stop`              | string or array              | no       | example: ["test"]                         |
| `stream`            | boolean                      | no       | `true`, `false` (default)                 |
| `temperature`       | float                        | no       | Float number between 0.0 and 2.0.         |
| `top_p`             | float                        | no       | Float number.                             |
| `user`              | string                       | no       | example: "test"                           |
| `ignore_eos`        | boolean                      | no       | `true`, `false` (default)                 |

Example request using curl

```
curl -X POST https://my.sample.endpoint.com/v1/chat/completions \
  -H "Content-type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is deep learning?"
      }
    ],
    "logprobs":true,
    "top_logprobs":1,
    "max_tokens":256
  }'
```

## Response Schema

By default, streaming is not enabled. The response is returned as application/json content-type:

| Field Name | Field Type                 | Always Returned | Possible Values                                        | 
|------------|----------------------------|-----------------|--------------------------------------------------------|
| `id`       | string                     | yes             | Example: "chatcmpl-0"                                  |
| `object`   | string                     | yes             | "chat.completion", "chat.completion.chunk"             |
| `created`  | int                        | yes             | The timestamp of when the chat completion was created. |
| `choices`  | array of [Choice](#choice) | yes             | See the [Choice](#choice) documentation                |
| `usage`    | [Usage](#usage)            | yes             | See the [Usage](#usage) documentation                  |

Example response:

```
{
   "id":"chatcmpl-0",
   "object":"chat.completion",
   "created":1711997860,
   "choices":[
      {
         "index":0,
         "message":{
            "role":"assistant",
            "content":"Hello! I\\'m here to help you with any questions you may have. Deep learning is a subfield of machine learning that involves the use"
         },
         "logprobs":null,
         "finish_reason":"length"
      }
   ],
   "usage":{
      "prompt_tokens":33,
      "completion_tokens":30,
      "total_tokens":63
   }
}
```

Chat Completions API supports streaming, and the response format for streaming differs from the response format for non-streaming.

To use streaming, set `"stream": true`.

The response is returned token by token as application/jsonlines content-type:

| Field Name  | Field Type                             | Always Returned | Possible Values                                        |
|-------------|----------------------------------------|-----------------|--------------------------------------------------------|
| `id`        | string                                 | yes             | Example: "chatcmpl-0"                                  |
| `object`    | string                                 | yes             | "chat.completion", "chat.completion.chunk"             |
| `created`   | int                                    | yes             | The timestamp of when the chat completion was created. |
| `choices`   | array of [StreamChoice](#streamchoice) | yes             | See the [StreamChoice](#streamchoice) documentation    |

Example response:

```
{
  "id": "chatcmpl-0", 
  "object": "chat.completion.chunk", 
  "created": 1712792433, 
  "choices": [
    {
      "index": 0, 
      "delta": {"content": " Oh", "role": "assistant"}, 
      "logprobs": [
        {
          "content": [
            {
              "token": " Oh", 
              "logprob": -4.499478340148926, 
              "bytes": [32, 79, 104], 
              "top_logprobs": [
                {
                  "token": -4.499478340148926, 
                  "logprob": -4.499478340148926, 
                  "bytes": [32, 79, 104]
                }
              ]
            }
          ]
        }
      ], 
      "finish_reason": null
    }
  ]
}
...
{
  "id": "chatcmpl-0", 
  "object": "chat.completion.chunk", 
  "created": 1712792436, 
  "choices": [
    {
      "index": 0, 
      "delta": {"content": " assist"}, 
      "logprobs": [
        {
          "content": [
            {
              "token": " assist", 
              "logprob": -1.019672155380249, 
              "bytes": [32, 97, 115, 115, 105, 115, 116], 
              "top_logprobs": [
                {
                  "token": -1.019672155380249, 
                  "logprob": -1.019672155380249, 
                  "bytes": [32, 97, 115, 115, 105, 115, 116]
                }
              ]
            }
          ]
        }
      ], 
      "finish_reason": "length"
    }
  ]
}
```

## API Object Schemas

The following sections describe each of the request or response objects in more detail.

### Message

The message object represents a single message of the conversation.
It contains the following fields:

| Field Name | Type                       | Description                | Example                                 |
|------------|----------------------------|----------------------------|-----------------------------------------|
| `role`     | string enum                | The role of the message    | "system", "user", "assistant"           |
| `content`  | string or array of objects | The content of the message | Example: "You are a helpful assistant." |

Example:

```
{
    "role":"system",
    "content":"You are a helpful assistant."
}
```

#### Vision/Image Support

You can specify an image as part of the content when using a vision language model.
Image data can either be specified as a url, or via a base64 encoding of the image data.

Example:

```
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What is this an image of?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "<base64 encoded image data | image url>"
            }
        }
    ]
}
```

We recommend that you use the base64 encoding to ensure no network failures occur when retrieving the image within the endpoint.
Network calls to fetch images can increase latency and introduce another failure point.

### Choice

The choice object represents a chat completion choice.
It contains the following fields:

| Field Name      | Type                  | Description                                       | Example                                    |
|-----------------|-----------------------|---------------------------------------------------|--------------------------------------------|
| `index`         | int                   | The index of the choice                           | 0                                          |
| `message`       | [Message](#message)   | A chat completion message generated by the model. | See the [Message](#message) documentation  |
| `logprobs`      | [Logprobs](#logprobs) | The log probability of the token                  | See the [Logprobs](#logprob) documentation |
| `finish_reason` | string enum           | The reason the model stopped generating tokens    | "length", "eos_token", "stop_sequence"     |

Example:

```
{
    "index":0,
    "message":{
       "role":"assistant",
       "content":"Hello! I\\'m here to help you with any questions you may have. Deep learning is a subfield of machine learning that involves the use"
    },
    "logprobs":null,
    "finish_reason":"length"
}
```

### StreamChoice

The choice object represents a chat completion choice.
It contains the following fields:

| Field Name      | Type                  | Description                                                    | Example                                     |
|-----------------|-----------------------|----------------------------------------------------------------|---------------------------------------------|
| `index`         | int                   | The index of the choice                                        | 0                                           |
| `delta`         | [Message](#message)   | A chat completion delta generated by streamed model responses. | See the [Message](#message) documentation   | 
| `logprobs`      | [Logprobs](#logprobs) | The log probability of the token                               | See the [Logprobs](#logprobs) documentation |
| `finish_reason` | string enum           | The reason the model stopped generating tokens                 | "length", "eos_token", "stop_sequence"      |

Example:

```
{
  "index": 0, 
  "delta": {"content": " Oh", "role": "assistant"}, 
  "logprobs": [
    {
      "content": [
        {
          "token": " Oh", 
          "logprob": -4.499478340148926, 
          "bytes": [32, 79, 104], 
          "top_logprobs": [
            {
              "token": -4.499478340148926, 
              "logprob": -4.499478340148926, 
              "bytes": [32, 79, 104]
            }
          ]
        }
      ]
    }
  ]
}
```

### Logprobs

Log probability information for the choice. This is only available when enable logprobs.
You must specify `logprobs=true` and set `top_logprobs` to a positive integer in the input `parameters`.

It contains a single field "content". Inside content, it contains the following fields:

| Field Name     | Type                               | Description                                                | Example                                         |
|----------------|------------------------------------|------------------------------------------------------------|-------------------------------------------------|
| `token`        | string                             | The token generated.                                       | " Oh"                                           |
| `logprob`      | float                              | The log probability of the generated token.                | -4.499478340148926                              |
| `bytes`        | array of float                     | The bytes representation of the token.                     | [32, 79, 104]                                   |
| `top_logprobs` | array of [TopLogprob](#toplogprob) | Array of the most likely tokens and their log probability. | See the [TopLogprob](#toplogprob) documentation |

Example:

```
{
   "content":[
      {
         "token":" ",
         "logprob":-4.768370445162873e-07,
         "bytes":[32],
         "top_logprobs":[
            {
               "token":" ",
               "logprob":-4.768370445162873e-07,
               "bytes":[32]
            }
         ]
      },
      {
         "token":" Deep",
         "logprob":-1.1869784593582153,
         "bytes":[32,68,101,101,112],
         "top_logprobs":[
            {
               "token":" Deep",
               "logprob":-1.1869784593582153,
               "bytes":[32,68,101,101,112]
            }
         ]
      }
   ]
}
```

### TopLogprob

Top log probability information for the choice.
It contains the following fields:

| Field Name     | Type                               | Description                                                | Example                                         |
|----------------|------------------------------------|------------------------------------------------------------|-------------------------------------------------|
| `token`        | string                             | The token generated.                                       | " Oh"                                           |
| `logprob`      | float                              | The log probability of the generated token.                | -4.499478340148926                              |
| `bytes`        | array of float                     | The bytes representation of the token.                     | [32, 79, 104]                                   |

```
{
    "token":" Deep",
    "logprob":-1.1869784593582153,
    "bytes":[32,68,101,101,112]
}
```

### Usage

Usage statistics for the completion request.
It contains the following fields:

| Field Name          | Type | Description                                                       | Example |
|---------------------|------|-------------------------------------------------------------------|---------|
| `completion_tokens` | int  | Number of tokens in the generated completion.                     | 33      |
| `prompt_tokens`     | int  | Number of tokens in the prompt.                                   | 100     |
| `total_tokens`      | int  | Total number of tokens used in the request (prompt + completion). | 133     |

Example:

```
"usage":{
    "prompt_tokens":33,
    "completion_tokens":100,
    "total_tokens":133
}
```
