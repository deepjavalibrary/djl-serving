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
from dataclasses import field, dataclass
from typing import List, Optional, Callable, Any, Dict, Union


class Token(object):
    """
    This class represents the token that comes to the output.
    """

    def __init__(self,
                 id: int,
                 text: str,
                 log_prob: float = None,
                 special_token: bool = None):
        """
        Initialize a Token

        :param id: token id in tokenizer
        :param text: the decoded text
        :param log_prob: log probability for the token
        :param special_token: if this token is special token
        """
        self.id = id
        self.text = text
        self.log_prob = log_prob
        self.special_token = special_token
        self.request_id = None

    def as_dict(self):
        output = {"id": self.id, "text": self.text, "log_prob": self.log_prob}
        if self.special_token:
            output["special_token"] = self.special_token
        return output

    def as_tgi_dict(self):
        output = {"id": self.id, "text": self.text, "logprob": self.log_prob}
        if self.special_token:
            output["special"] = self.special_token
        return output


class Iterator:

    def __init__(self):
        self._index = 0

    def get_index(self):
        return self._index

    def next_index(self):
        next_index = self._index
        self._index += 1
        return next_index


@dataclass
class Sequence:
    """Dataclass to store the generated tokens details of all the sequences in the request.
    Attributes:
        tokens: generated tokens of the sequence.
        top_tokens: top tokens of the sequence.
        cumulative_log_prob: cumulative log probability of the sequence.
        finish_reason: finish reason of the sequence.
        stop_reason: stop reason of the sequence.
        first_token: first token of the sequence.
        last_token: last token of the sequence.
    """
    tokens: List[Token] = field(default_factory=lambda: [])
    top_tokens: Optional[List[List[Token]]] = field(default_factory=lambda: [])
    cumulative_log_prob: float = 0.0
    finish_reason: str = None
    _last_token_index: Optional[int] = None
    stop_reason: Optional[str] = None
    _tokens_iterator: Optional[Iterator] = field(init=False, default=None)
    _top_tokens_iterator: Optional[Iterator] = field(init=False, default=None)

    def __post_init__(self):
        self._tokens_iterator = Iterator()
        self._top_tokens_iterator = Iterator()

    def set_next_token(self, token: Token, is_last_token: bool = False):
        self.tokens.append(token)
        if is_last_token:
            self._last_token_index = len(self.tokens) - 1

    def set_next_top_tokens(self, top_tokens: List[Token]):
        self.top_tokens.append(top_tokens)

    def has_next_token(self):
        return self._tokens_iterator.get_index() < len(self.tokens)

    def has_next_top_tokens(self):
        return self._top_tokens_iterator.get_index() < len(self.top_tokens)

    def get_next_token(self) -> (Token, bool, bool):
        if self.has_next_token():
            index = self._tokens_iterator.next_index()
            first_token = index == 0
            last_token = index == self._last_token_index
            return self.tokens[index], first_token, last_token
        return None, False, False

    def get_next_top_tokens(self):
        """Returns the next list of top tokens from the top_tokens list, or None if all have been iterated."""
        if self.has_next_top_tokens():
            index = self._top_tokens_iterator.next_index()
            return self.top_tokens[index]
        return None


@dataclass
class RequestInput:
    """Base class for all request inputs.
    Attributes:
        request_id: The request ID.
        output_formatter: Output formatter of the request
        parameters: parameters in the request payload
    """
    request_id: int
    output_formatter: Union[Callable, str] = None
    parameters: Dict = field(default_factory=lambda: {})
    tgi_compat: bool = False


@dataclass
class TextInput(RequestInput):
    """Request input for text generation.
    Attributes:
        input_text: The input text.
        input_ids: The input tokens ids.
        adapters: adapter used for the request.
        tokenizer: tokenizer used for the request.
    """
    input_text: str = None
    input_ids: List[int] = field(default_factory=lambda: [])
    adapters: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def prompt_tokens_length(self) -> int:
        return len(self.input_ids)


@dataclass
class RequestOutput:
    """Base class for all request outputs.
    Attributes:
        request_id: The request ID.
        input: The request input.
        finished: Whether the request is finished.
    """
    request_id: int
    input: RequestInput
    finished: bool = False


@dataclass
class TextGenerationOutput(RequestOutput):
    """Request output for text generation.
    Attributes:
        sequences: generated sequences. If the user use beam search or best_of,multiple sequences could be generated.
        best_sequence_index: index of the best sequence.
        prompt_tokens_details: prompt tokens details such as their log_probs and token_ids.
    """
    sequences: dict[int, Sequence] = field(default_factory=lambda: {})
    best_sequence_index: int = 0
    prompt_tokens_details: List[Token] = field(default_factory=lambda: [])

    def set_next_token(self,
                       token: Token,
                       sequence_index=0,
                       is_last_token: bool = False):
        """Adds token to the given index sequence. If not given, adds to the first sequence.

        :param is_last_token: whether the token is the last token of the sequence.
        :param token: token to add to the given sequence.
        :param sequence_index: index of the sequence to add the token to.
        """
        if sequence_index not in self.sequences:
            self.sequences[sequence_index] = Sequence()
        self.sequences[sequence_index].set_next_token(token, is_last_token)

    def set_next_top_tokens(self, top_tokens: List[Token], sequence_index):
        self.sequences[sequence_index].top_tokens.append(top_tokens)

    def set_best_sequence_index(self, sequence_index):
        self.best_sequence_index = sequence_index

    def set_finish_reason(self, finish_reason: str, sequence_index=0):
        """Sets the finish reason for the given sequence index. If not given, sets the finish reason for the first.

        :param finish_reason: finish reason for the sequence.
        :param sequence_index: index of the sequence to set the finish reason.
        """
        self.sequences[sequence_index].finish_reason = finish_reason

    def get_tokens_as_dict(self, sequence_index=0):
        """Returns the tokens of the given sequence index as a dictionary.
        If not given, returns the tokens of the first sequence index as a dictionary.

        :param sequence_index: index of the sequence to get the tokens from.
        :return: tokens of the given sequence index as a dictionary.
        """
        tokens = []
        for token in self.sequences[sequence_index].tokens:
            if self.input.tgi_compat:
                tokens.append(token.as_tgi_dict())
            else:
                tokens.append(token.as_dict())
        return tokens

    def get_prompt_tokes_as_dict(self):
        """Returns the prompt tokens as a dictionary.

        :return: prompt tokens as a dictionary.
        """
        tokens = []
        for token in self.prompt_tokens_details:
            if self.input.tgi_compat:
                tokens.append(token.as_tgi_dict())
            else:
                tokens.append(token.as_dict())
        return tokens
