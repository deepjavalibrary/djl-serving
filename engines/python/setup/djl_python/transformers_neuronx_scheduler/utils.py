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
