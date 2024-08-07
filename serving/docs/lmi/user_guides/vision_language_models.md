# Vision Language Models in LMI

> [!WARNING]
> Vision Language Model support is currently experimental in v0.29.0.
> LMI currently only supports using VLMs with the lmi-dist and vllm backends.

Starting with v0.29.0, LMI supports deploying the following types of Vision Language Models:

* llava (e.g. llava-hf/llava-1.5-7b-hf)
* llava_next (e.g. llava-hf/llava-v1.6-mistral-7b-hf)
* phi3_v (e.g. microsoft/Phi-3-vision-128k-instruct)
* paligemma (e.g. google/paligemma-3b-mix-224)
* chameleon (facebook/chameleon-7b etc.)
* fuyu (adept/fuyu-8b etc.)

## Request Format

We currently support the OpenAI Chat Completions schema for invoking Vision Language Models.
You can read more about the supported format in the [chat completions doc](chat_input_output_schema.md).

## Deploying with LMI

Deploying Vision Language Models with LMI is very similar to deploying Text Generation Models.
There is an additional, optional config that is exposed, `option.image_placeholder_token` that we recommend you set.
This config specifies the image placeholder token, which is then used by the model's processor and tokenizer to determine where to place the image content in the prompt.
We recommend you set this value explicitly because it is challenging to determine from the model artifacts.

Example SageMaker deployment code:

```python
from sagemaker.djl_inference import DJLModel

model = DJLModel(
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    env={
        "OPTION_IMAGE_PLACEHOLDER_TOKEN": "<image>",
    }
)

messages = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is this an image of?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://resources.djl.ai/images/dog_bike_car.jpg"
                    }
                }
            ]
        }
    ]
}

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g6.12xlarge",
)

response = predictor.predict(messages)
```