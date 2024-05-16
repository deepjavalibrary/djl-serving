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

from djl_python.rolling_batch.rolling_batch import Token


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
    finish_reason: str = False
    last_token: bool = False
    first_token: bool = True
    stop_reason: Optional[str] = None
    _tokens_iterator: Optional[Iterator] = field(init=False, default=None)
    _top_tokens_iterator: Optional[Iterator] = field(init=False, default=None)

    def __post_init__(self):
        self._tokens_iterator = Iterator()
        self._top_tokens_iterator = Iterator()

    def set_next_token(self, token: Token):
        self.tokens.append(token)

    def set_next_top_tokens(self, top_tokens: List[Token]):
        self.top_tokens.append(top_tokens)

    def has_next_token(self):
        return (self._tokens_iterator.get_index() + 1) < len(self.tokens)

    def has_next_top_tokens(self):
        return (self._top_tokens_iterator.next_index() + 1) < len(
            self.top_tokens)

    def get_next_token(self):
        if self.has_next_token():
            index = self._tokens_iterator.next_index()
            return self.tokens[index]
        return None

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
    parameters: Dict = field(default_factory=lambda: [])


@dataclass
class TextInput(RequestInput):
    """Request input for text generation.
    Attributes:
        input_text: The input text.
        input_ids: The input tokens ids.
        adapter: adapter used for the request.
        tokenizer: tokenizer used for the request.
    """
    input_text: str = None
    input_ids: List[int] = field(default_factory=lambda: [])
    adapter: Optional[Any] = None
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
    prompt_tokens_details: List[dict] = field(default_factory=lambda: {})

    def set_next_token(self, token: Token, sequence_index=0):
        if sequence_index not in self.sequences:
            self.sequences[sequence_index] = Sequence()
        self.sequences[sequence_index].set_next_token(token)

    def set_next_top_tokens(self, top_tokens: List[Token], sequence_index):
        self.sequences[sequence_index].top_tokens.append(top_tokens)

    def set_best_sequence_index(self, sequence_index):
        self.best_sequence_index = sequence_index
