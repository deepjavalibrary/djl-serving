# Tool Calling Support in LMI

Tool calling is currently supported in LMI through the [vLLM](vllm_user_guide.md) backend only.

Details on vLLM's tool calling support can be found [here](https://docs.vllm.ai/en/v0.9.0.1/features/tool_calling.html#how-to-write-a-tool-parser-plugin).

To enable tool calling in LMI, you must set the following environment variable configurations:

```
OPTION_ROLLING_BATCH=vllm
OPTION_ENABLE_AUTO_TOOL_CHOICE=true
OPTION_TOOL_CALL_PARSER=<parser_name>
```

You can find built-in tool call parsers [here](https://docs.vllm.ai/en/v0.9.0.1/features/tool_calling.html#automatic-function-calling).

Additionally, you must provide a chat template that supports tool parsing.
You can specify a specific chat template using the `OPTION_CHAT_TEMPLATE=<path/to/template>` environment variable.
We recommend that you provide this chat template as part of the model artifacts, and only provide a single chat template.

## Deploying with LMI

```python
from sagemaker.djl_inference import DJLModel

image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126"
role = "my-iam-role"

model = DJLModel(
    image_uri=image_uri,
    role=role,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "HF_TOKEN": "***",
        "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",
        "OPTION_TOOL_CALL_PARSER": "llama3_json",
        "OPTION_ROLLING_BATCH": "vllm",
    }
)

messages =  {
    "messages": [
        {
            "role": "user",
            "content": "Hi! How are you doing today?"
        }, 
        {
            "role": "assistant",
            "content": "I'm doing well! How can I help you?"
        }, 
        {
            "role": "user",
            "content": "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
        }
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                            "string",
                        "description":
                            "The city to find the weather for, e.g. 'San Francisco'"
                    },
                    "state": {
                        "type":
                            "string",
                        "description":
                            "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"
                    },
                    "unit": {
                        "type": "string",
                        "description":
                            "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city", "state", "unit"]
            }
        }
    }],
    "tool_choice": {
        "type": "function",
        "function": {
            "name": "get_current_weather"
        }
    },
}

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g6.12xlarge",
)

response = predictor.predict(messages)

print(response)
```
```
# sample output
{
    'id': 'chatcmpl-140510529166160',
    'object': 'chat.completion',
    'created': 1743550385,
    'choices': [
        {
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'chatcmpl-tool-140510529166160',
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'arguments': '{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}'
                        }
                    }
                ]
            },
            'logprobs': None,
            'finish_reason': 'eos_token'
        }
    ],
    'usage': {
        'prompt_tokens': 338, 
        'completion_tokens': 20, 
        'total_tokens': 358
    }
}
```
