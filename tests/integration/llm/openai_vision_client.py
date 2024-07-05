import argparse
import base64
import sys

import requests
from openai import OpenAI

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8080/invocations"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)


def call_chat_completion_api(image: str):

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
                    "url": f"{image}"
                },
            },
        ],
    }]

    chat_completion_with_image = client.chat.completions.create(
        messages=sample_messages,
        model="",
    )

    return chat_completion_with_image


def get_image_url(image_url_type: str, image: str):
    if image_url_type == "base64":
        if image.startswith("http"):
            with requests.get(image_url) as response:
                response.raise_for_status()
                image_base64 = base64.b64encode(
                    response.content).decode('utf-8')
        else:
            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read())
        return f"data:image/jpeg;base64,{image_base64}"
    else:
        return image


def run(raw_args):
    parser = argparse.ArgumentParser(description="OpenAI VLM API client")
    parser.add_argument("image_url_type",
                        type=str,
                        choices=["url", "base64"],
                        default="url",
                        help="image url type")
    parser.add_argument(
        "image",
        type=str,
        default="https://resources.djl.ai/images/dog_bike_car.jpg",
        help="image http url or local path")

    global args
    args = parser.parse_args(args=raw_args)

    image_url = get_image_url(args.image_url_type, args.image)
    result = call_chat_completion_api(image_url)
    print(f"OpenAI vision client result {result}")


if __name__ == "__main__":
    run(sys.argv[1:])
