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
from typing import Optional, Union, List, Dict

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class ChatProperties(BaseModel):
    """
    Chat input parameters for chat completions API.
    See https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: List[Dict[str, str]]
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

    @field_validator('messages', mode='before')
    def validate_messages(
            cls, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if messages is None:
            return None

        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError(
                    "When passing chat dicts as input, each dict must have a 'role' and 'content' key."
                )
        return messages

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
