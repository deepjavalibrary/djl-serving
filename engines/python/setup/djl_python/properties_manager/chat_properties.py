from typing import Optional, Union, List, Dict

from pydantic.v1 import BaseModel, Field, validator, root_validator


class ChatProperties(BaseModel):
    """
    Chat input parameters for chat completions API.
    See https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: List[Dict[str, str]]
    model: Optional[str]  # UNUSED
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[dict] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int]  # Currently only support 1
    max_new_tokens: Optional[int] = Field(alias="max_tokens")
    n: Optional[int] = 1  # Currently only support 1
    presence_penalty: Optional[float] = 0
    seed: Optional[int]
    stop_sequences: Optional[Union[str, list]] = Field(alias="stop")
    temperature: Optional[int] = 1
    top_p: Optional[int] = 1
    user: Optional[str]

    @validator('messages', pre=True)
    def validate_messages(
            cls, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if messages is None:
            return messages

        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError(
                    "When passing chat dicts as input, each dict must have a 'role' and 'content' key."
                )
        return messages

    @validator('frequency_penalty', pre=True)
    def validate_frequency_penalty(cls, frequency_penalty: float) -> float:
        if frequency_penalty is None:
            return frequency_penalty

        frequency_penalty = float(frequency_penalty)
        if frequency_penalty < -2.0 or frequency_penalty > 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0.")
        return frequency_penalty

    @validator('top_logprobs', pre=True)
    def validate_top_logprobs(cls, top_logprobs: float) -> float:
        if top_logprobs is None:
            return top_logprobs

        top_logprobs = int(top_logprobs)
        if top_logprobs < 0 or top_logprobs > 20:
            raise ValueError("top_logprobs must be between 0 and 20.")
        return top_logprobs

    @validator('presence_penalty', pre=True)
    def validate_presence_penalty(cls, presence_penalty: float) -> float:
        if presence_penalty is None:
            return presence_penalty

        presence_penalty = float(presence_penalty)
        if presence_penalty < -2.0 or presence_penalty > 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0.")
        return presence_penalty

    @validator('temperature', pre=True)
    def validate_temperature(cls, temperature: float) -> float:
        if temperature is None:
            return temperature

        temperature = float(temperature)
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2.")
        return temperature
