#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import base64
from io import BytesIO
from typing import Union

import requests
from PIL import Image

# TODO: Image token differs for each VLM model.
# Including model_config becomes easier once parse_input refactor PR is done.


def get_image_text_prompt(prompt_text: str) -> str:
    # TODO: image token str must be decoded from image_token_id in serving.properties. Change it after refactor PR.
    image_token_str = '<image>'

    return f"{image_token_str}\n{prompt_text}"


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image)))


def fetch_image_from_url(image_url: str) -> Image.Image:
    # TODO: Add configurable timeout, by using an env or serving.properties, from properties.py
    # TODO: add validation for http url
    # TODO: Now, we always assume, it is an image format, it could also be pixel numpy file or image features file (pt)
    # Fetches the image from the http url
    with requests.get(url=image_url) as response:
        response.raise_for_status()
        image_raw = response.content
    # Opens the image using pillow, but it does not load the model into memory yet
    # (image.load()), as some frameworks like vllm does it anyway.
    image = Image.open(BytesIO(image_raw))
    return image


def fetch_image(image_url: str) -> Image.Image:
    if image_url.startswith("http"):
        return fetch_image_from_url(image_url)
    elif image_url.startswith("data:image"):
        _, image_base64 = image_url.split(",", 1)
        return load_image_from_base64(image_base64)
    else:
        raise ValueError("Invalid image url")


# Use base64 encoded image in the payload
def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode('utf-8')
    return base64_image
