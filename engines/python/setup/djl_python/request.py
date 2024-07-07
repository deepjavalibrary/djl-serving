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
import inspect
from typing import Union, Callable, Any, List, Dict

from djl_python.output_formatter import get_output_formatter, adapt_legacy_output_formatter
from djl_python.request_io import Token, TextGenerationOutput, TextInput, RequestOutput, RequestInput
from djl_python.utils import wait_till_generation_finished


class Request(object):
    """
    This class represents each request that comes to the handler.

    In rolling batch, handler is called for each forward function.
    So this class represents the states of each request until the
    last token is generated.

    """

    def __init__(self, request_input: TextInput = None):
        """
        Initialize a request

        :param id: request id
        """

        #TODO: Remove some of these redundant attributes and
        # use request_input and request_output wherever necessary.
        self.id = request_input.request_id
        self.request_input = request_input
        self.input_text = request_input.input_text
        self.last_token = False
        self.adapter = request_input.adapters

        # server parameters may not be set, if custom input formatter is used.
        if not self.request_input.server_parameters:
            self.request_input.server_parameters = self.request_input.parameters.copy(
            )
        self.parameters = self.request_input.server_parameters

        # output formatter
        request_input.output_formatter = self.parameters.pop(
            "output_formatter", request_input.output_formatter)
        # stream parameter is only used for determining the output.
        stream = self.parameters.pop("stream", False)
        # details is only used in output formatter for rolling batch
        self.parameters.pop("details", False)
        self.output_formatter, self.content_type = get_output_formatter(
            request_input.output_formatter, stream, request_input.tgi_compat)
        request_input.output_formatter = self.output_formatter
        self.legacy_formatter = self._is_output_formatter_legacy()

        self.request_output = TextGenerationOutput(request_id=self.id,
                                                   input=self.request_input)
        self.next_token_str = ""

    def _is_output_formatter_legacy(self):
        signature_parameters = list(
            inspect.signature(self.output_formatter).parameters.values())
        return signature_parameters[0].annotation not in [
            RequestOutput, TextGenerationOutput
        ]

    def __repr__(self):
        return f"<Request id: {self.id} Input {self.input_text} Parameters {self.parameters} Finished {self.last_token}>"

    def set_next_token(self,
                       next_token: Union[Token, str],
                       last_token: bool = False,
                       finish_reason: str = None,
                       prompt_tokens_details: List[Token] = None):
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
        self.request_output.set_next_token(next_token,
                                           is_last_token=last_token,
                                           finish_reason=finish_reason)
        self.request_output.prompt_tokens_details = prompt_tokens_details
        self.last_token = last_token
        if last_token:
            self.request_output.finished = True

    def get_next_token(self) -> str:
        """
        Gets the token generated for the request.

        :return: next_token
        """
        if self.next_token_str:
            return self.next_token_str
        else:
            # TODO: Remove this support when all of our customers onboard.
            if self.legacy_formatter:
                self.next_token_str = adapt_legacy_output_formatter(
                    self.request_output)
            elif wait_till_generation_finished(
                    self.request_output.input.parameters):
                # there is no need for iterators for best_of and num_beams.
                self.next_token_str = self.output_formatter(
                    self.request_output)
            else:
                best_sequence = self.request_output.sequences[
                    self.request_output.best_sequence_index]
                while best_sequence.has_next_token():
                    self.next_token_str += self.output_formatter(
                        self.request_output)
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
