import base64
import unittest

from openai import OpenAI
from transformers import AutoTokenizer

from djl_python.chat_completions.chat_utils import parse_chat_completions_request
from djl_python.multimodal.utils import encode_image_base64_from_url

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)


class TestLmiDist(unittest.TestCase):

    def test_open_ai_format_parse(self):
        image_url = "https://resources.djl.ai/images/dog_bike_car.jpg"
        image_base64 = encode_image_base64_from_url(image_url=image_url)
        sample_messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }]
        sample_input_map = {'messages': sample_messages, 'model': ""}
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-34b-hf",
                                                  use_fast=False)
        inputs, params = parse_chat_completions_request(sample_input_map,
                                                        is_rolling_batch=True,
                                                        tokenizer=tokenizer)
        print(inputs)
        images = params.pop("images", None)
        for image in images:
            print(image)
        print(params)
