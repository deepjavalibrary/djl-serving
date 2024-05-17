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
from typing import Union, Callable

from djl_python.output_formatter import get_output_formatter, _json_output_formatter, sse_response_formatter
from djl_python.request_io import Token


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self,
                 id: int,
                 input_text: str,
                 parameters: dict,
                 details: bool = False,
                 input_ids: list = [],
                 adapter=None,
                 output_formatter: Union[str, Callable] = None,
                 tgi_compat: bool = False):
        """
        Initialize a request

        :param id: request id
        :param input_text: request's input text
        :param parameters: list of parameters
        :param details: whether to include details
        :param input_ids: request's input ids
        :param adapter: list of adapters
        :param output_formatter: output formatter function (for example,
            _json_output_formatter, _jsonlines_output_formatter, or user provided function
        """
        self.id = id
        self.input_text = input_text
        self.parameters = parameters
        self.original_params = parameters.copy()
        self.details = details
        self.adapter = adapter
        self.input_ids = input_ids
        self.next_token_str = ""
        self.first_token = True
        self.last_token = False
        self.token_cache = None
        self.generated_tokens = []
        self.decoder_input_details = parameters.get("decoder_input_details",
                                                    False)
        self.tgi_compat = tgi_compat
        if self.details:
            self.token_cache = []
        self.full_text_prefix = input_text if parameters.pop(
            "return_full_text", False) else ""

        # output formatter
        stream = parameters.pop("stream", False)
        self.output_formatter, self.content_type = get_output_formatter(
            output_formatter, stream, self.tgi_compat)

    def __repr__(self):
        return f"<Request id: {self.id} Input {self.input_text} Parameters {self.parameters} Finished {self.last_token}>"

    def set_next_token(self,
                       next_token: Union[Token, str],
                       last_token: bool = False,
                       finish_reason: str = None,
                       prompt_tokens_details: list[dict] = None):
        """
        Sets the newly generated token.
        If the function is called for multiple times, we will append tokens to the token string.

        :param next_token: next token to be set.
        :param last_token: whether this token is the last of the sequence.
        :param finish_reason: what reason made the generation ends. Current options:
            length: end because max_output_token size reached
            eos_token: End of sequence token found
            stop_sequence: Preset stop sequence token found
        :param prompt_tokens_details: prompt tokens details when parameter decoder_input_details is true.
        """
        if isinstance(next_token, str):
            next_token = Token(-1, next_token)
        next_token.request_id = self.id
        if self.token_cache is not None:
            if self.tgi_compat:
                self.token_cache.append(next_token.as_tgi_dict())
            else:
                self.token_cache.append(next_token.as_dict())
        self.generated_tokens.append(next_token.text)
        details_dict = {}
        # making detailed information captured for each token generation
        if self.details:
            details_dict["finish_reason"] = finish_reason
            details_dict["tokens"] = self.token_cache
            details_dict["generated_tokens"] = len(self.token_cache)
            details_dict["inputs"] = self.input_text
            details_dict["parameters"] = self.original_params
            details_dict["prompt_tokens"] = len(self.input_ids)
        # Special handling for error case
        elif finish_reason == "error":
            details_dict["finish_reason"] = finish_reason
        if self.output_formatter == _json_output_formatter or self.output_formatter == sse_response_formatter:
            details_dict["tgi_compat"] = self.tgi_compat
        generated_text = self.full_text_prefix
        if last_token:
            generated_text = generated_text + ''.join(self.generated_tokens)
            if self.decoder_input_details:
                details_dict["prompt_tokens_details"] = prompt_tokens_details
        if self.output_formatter is None:
            self.next_token_str += next_token.text
        else:  # output only supports size one now
            self.next_token_str += self.output_formatter(
                next_token, self.first_token, last_token, details_dict,
                generated_text, self.id)
        self.last_token = last_token
        self.first_token = False

    def get_next_token(self) -> str:
        """
        Gets the token generated for the request.

        :return: next_token
        """
        return self.next_token_str

    def reset_next_token(self):
        """
        Reset the next token.
        """
        self.next_token_str = ""

    def is_last_token(self) -> bool:
        """
        Whether the generated token is the last one

        :return: whether last token of the sequence.
        """
        return self.last_token

    def get_content_type(self) -> str:
        """
        Content type of this particular request

        :return: content type
        """
        return self.content_type
