#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
from enum import Enum
from typing import Optional


class FinishReason(str, Enum):
    FINISH_REASON_LENGTH = 0
    FINISH_REASON_EOS_TOKEN = 1
    FINISH_REASON_STOP_SEQUENCE = 2

    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


class Generation(object):

    def __init__(self, request_id, prefill_tokens, token_id, token_logprob,
                 token_text, token_is_special, generated_text) -> None:
        self.request_id = request_id
        self.prefill_tokens = prefill_tokens
        self.token_id = token_id
        self.token_logprob = token_logprob
        self.token_text = token_text
        self.token_is_special = token_is_special
        self.generated_text = generated_text


class GeneratedText(object):

    def __init__(self, text: str, generated_tokens: int,
                 finish_reason: FinishReason, seed: Optional[int]):
        self.text = text
        self.generated_tokens = generated_tokens
        self.finish_reason = finish_reason
        self.seed = seed


class TokenDecoder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0
        self.all_input_ids = []

    def _decode_token(self) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:self.read_offset],
            skip_special_tokens=False)
        new_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:], skip_special_tokens=False)
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.all_input_ids)
            return new_text
        else:
            return ""

    def decode(self, token_id: int):
        self.all_input_ids.append(token_id)
        return self._decode_token()
