# Vision Language Models in LMI


LMI supports deploying the following types of Vision Language Models through the vLLM backend:

* llava (e.g. llava-hf/llava-1.5-7b-hf)
* llava_next (e.g. llava-hf/llava-v1.6-mistral-7b-hf)
* phi3_v (e.g. microsoft/Phi-3-vision-128k-instruct)
* paligemma (e.g. google/paligemma-3b-mix-224)
* chameleon (facebook/chameleon-7b etc.)
* fuyu (adept/fuyu-8b etc.)
* pixtral (mistralai/Pixtral-12B-2409)
* mLlama (meta-llama/Llama-3.2-11B-Vision-Instruct)

## Request Format

We currently support the OpenAI Chat Completions schema for invoking Vision Language Models.
You can read more about the supported format in the [chat completions doc](chat_input_output_schema.md).

## Deploying with LMI

Deploying Vision Language Models with LMI is very similar to deploying Text Generation Models.

There are some additional, optional configs that are exposed:
* `option.limit_mm_per_prompt`: For each multimodal plugin, limit how many input instances to allow for each prompt. Expects a comma-separated list of items, e.g.: `{"image": 16, "video": 2}` allows a maximum of 16 images and 2 videos per prompt. Defaults to 1 for each modality.

Example SageMaker deployment code:

```python
from sagemaker.djl_inference import DJLModel

model = DJLModel(
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    env={
        "OPTION_LIMIT_MM_PER_PROMPT": '{"image":2}',
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
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://resources.djl.ai/images/kitten.jpg"
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