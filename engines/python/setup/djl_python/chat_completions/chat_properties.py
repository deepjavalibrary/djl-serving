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
from typing import Optional, Union, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from PIL.Image import Image

from djl_python.multimodal.utils import fetch_image


class TextInput:
    text: str

    def __init__(self, text: str):
        self.text = text


class ImageInput:
    image: Image

    def __init__(self, image_url: str):
        self.image = fetch_image(image_url)


class Message(BaseModel):
    # This is needed because TextInput/ImageInput are not BaseModel instances
    # they don't really need to be, but we could figure out a way to avoid this if needed
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str
    content: List[
        Union[TextInput, ImageInput],
    ]

    @field_validator('content', mode='before')
    def validate_content(
        cls, contents: Union[str, List[Dict[str, Any]]]
    ) -> List[Union[TextInput, ImageInput]]:
        if isinstance(contents, str):
            return [TextInput(contents)]

        transformed_content = []
        for content in contents:
            if "type" not in content:
                raise ValueError(
                    "You must provide 'type' for each object when providing a list of objects as content"
                )
            content_type = content["type"]
            if content_type == "text":
                if "text" not in content:
                    raise ValueError(
                        "'text' type content must specify the 'text'")
                transformed_content.append(TextInput(content["text"]))
            elif content_type == "image_url":
                image_url = content.get("image_url", {})
                url = image_url.get("url")
                if url is None:
                    raise ValueError(
                        "image_url is not provided correctly. you must provide images as {'type': "
                        "'image_url', 'image_url': {'url': <value>}}")
                transformed_content.append(ImageInput(url))
        return transformed_content

    def get_tokenizer_inputs(self, image_token="<image>"):
        texts = []
        images = []
        for content in self.content:
            if isinstance(content, TextInput):
                texts.append(content.text)
            else:
                images.append(content.image)

        prompt_text = '\n'.join(texts)
        if len(images) > 0:
            prompt_text = f"{image_token}\n{prompt_text}"
        return {
            "role": self.role,
            "content": prompt_text,
        }

    def get_images(self) -> List[Image]:
        return [i.image for i in self.content if isinstance(i, ImageInput)]


class ChatProperties(BaseModel):
    """
    Chat input parameters for chat completions API.
    See https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: List[Message]
    model: Optional[str] = Field(default=None, exclude=True)  # Unused
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = Field(default=None, exclude=True)
    logprobs: Optional[bool] = Field(default=False, exclude=True)
    top_logprobs: Optional[int] = Field(default=None,
                                        serialization_alias="logprobs")
    max_tokens: Optional[int] = Field(default=None,
                                      serialization_alias="max_new_tokens")
    n: Optional[int] = Field(default=1, exclude=True)
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = Field(default=None, exclude=True)

    @field_validator('frequency_penalty', mode='before')
    def validate_frequency_penalty(
            cls, frequency_penalty: Optional[float]) -> Optional[float]:
        if frequency_penalty is None:
            return None

        frequency_penalty = float(frequency_penalty)
        if frequency_penalty < -2.0 or frequency_penalty > 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0.")
        return frequency_penalty

    @field_validator('logit_bias', mode='before')
    def validate_logit_bias(cls, logit_bias: Dict[str, float]):
        if logit_bias is None:
            return None

        for token_id, bias in logit_bias.items():
            if bias < -100.0 or bias > 100.0:
                raise ValueError(
                    "logit_bias value must be between -100 and 100.")
        return logit_bias

    @field_validator('top_logprobs')
    def validate_top_logprobs(cls, top_logprobs: int, info: ValidationInfo):
        if top_logprobs is None:
            return None

        if not info.data.get('logprobs'):
            return None

        top_logprobs = int(top_logprobs)
        if top_logprobs < 0 or top_logprobs > 20:
            raise ValueError("top_logprobs must be between 0 and 20.")
        return top_logprobs

    @field_validator('presence_penalty', mode='before')
    def validate_presence_penalty(
            cls, presence_penalty: Optional[float]) -> Optional[float]:
        if presence_penalty is None:
            return None

        presence_penalty = float(presence_penalty)
        if presence_penalty < -2.0 or presence_penalty > 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0.")
        return presence_penalty

    @field_validator('temperature', mode='before')
    def validate_temperature(cls,
                             temperature: Optional[float]) -> Optional[float]:
        if temperature is None:
            return None

        temperature = float(temperature)
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0 and 2.")
        return temperature
