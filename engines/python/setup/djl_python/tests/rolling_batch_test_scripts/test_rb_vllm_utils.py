import sys
import unittest
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Union
from collections import OrderedDict
from unittest import mock
from unittest.mock import Mock, MagicMock

from transformers import AutoTokenizer

import djl_python
from djl_python.output_formatter import _json_output_formatter
from djl_python.request import Request
from djl_python.request_io import TextGenerationOutput, TextInput, Sequence, Token
'''These Mock classes are in compliance with vllm RequestOutput version 0.4.2'''


@dataclass
class MockLogprob:
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


MockPromptLogprobs = List[Optional[Dict[int, MockLogprob]]]
MockSampleLogprobs = List[Dict[int, MockLogprob]]


class MockCompletionOutput:

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[MockSampleLogprobs],
        finish_reason: Optional[str] = None,
        stop_reason: Union[int, str, None] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason

    def finished(self) -> bool:
        return self.finish_reason is not None


class MockRequestOutput:

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        prompt_logprobs: Optional[MockPromptLogprobs],
        outputs: List[MockCompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished


example_request_output = [
    MockRequestOutput(
        request_id="test_request_id",
        prompt="I am a",
        prompt_token_ids=[1, 315, 837, 264],
        prompt_logprobs=[
            None, {
                315: MockLogprob(logprob=-4.37, rank=8, decoded_token='I'),
                422: MockLogprob(logprob=-1.25, rank=1, decoded_token='#')
            }, {
                837: MockLogprob(logprob=-2.62, rank=3, decoded_token='am'),
            }, {
                264: MockLogprob(logprob=-1.73, rank=1, decoded_token='a')
            }
        ],
        outputs=[
            MockCompletionOutput(index=1,
                                 text=' member',
                                 token_ids=[4292],
                                 cumulative_logprob=-4.2740092277526855,
                                 logprobs=[{
                                     4292:
                                     MockLogprob(logprob=-4.2740092277526855,
                                                 rank=4,
                                                 decoded_token=' member'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new')
                                 }],
                                 finish_reason=None,
                                 stop_reason=None),
            MockCompletionOutput(index=0,
                                 text=' consolid',
                                 token_ids=[22968],
                                 cumulative_logprob=-12.117759704589844,
                                 logprobs=[{
                                     22968:
                                     MockLogprob(logprob=-12.117759704589844,
                                                 rank=5308,
                                                 decoded_token=' consolid'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     17372:
                                     MockLogprob(logprob=-13.409988403320312,
                                                 rank=10489,
                                                 decoded_token=' crown'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new'),
                                 }],
                                 finish_reason=None,
                                 stop_reason=None)
        ],
        finished=False),
    MockRequestOutput(
        request_id="test_request_id",
        prompt="I am a",
        prompt_token_ids=[1, 315, 837, 264],
        prompt_logprobs=[
            None, {
                315: MockLogprob(logprob=-4.37, rank=8, decoded_token='I'),
                422: MockLogprob(logprob=-1.25, rank=1, decoded_token='#')
            }, {
                837: MockLogprob(logprob=-2.62, rank=3, decoded_token='am'),
            }, {
                264: MockLogprob(logprob=-1.73, rank=1, decoded_token='a')
            }
        ],
        outputs=[
            MockCompletionOutput(index=1,
                                 text=' member of',
                                 token_ids=[4292, 302],
                                 cumulative_logprob=-4.3041129764169455,
                                 logprobs=[{
                                     4292:
                                     MockLogprob(logprob=-4.2740092277526855,
                                                 rank=4,
                                                 decoded_token=' member'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new'),
                                 }, {
                                     302:
                                     MockLogprob(logprob=-0.03010374866425991,
                                                 rank=1,
                                                 decoded_token=' of'),
                                     235290:
                                     MockLogprob(logprob=-2.2026185989379883,
                                                 rank=1,
                                                 decoded_token='-'),
                                     578:
                                     MockLogprob(logprob=-2.2026185989379883,
                                                 rank=2,
                                                 decoded_token=' and')
                                 }],
                                 finish_reason=None,
                                 stop_reason=None),
            MockCompletionOutput(index=0,
                                 text=' consolidated',
                                 token_ids=[22968, 601],
                                 cumulative_logprob=-13.402491569519043,
                                 logprobs=[{
                                     22968:
                                     MockLogprob(logprob=-12.117759704589844,
                                                 rank=5308,
                                                 decoded_token=' consolid'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     17372:
                                     MockLogprob(logprob=-13.409988403320312,
                                                 rank=10489,
                                                 decoded_token=' crown'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new'),
                                 }, {
                                     601:
                                     MockLogprob(logprob=-1.2847318649291992,
                                                 rank=2,
                                                 decoded_token='ated'),
                                     1028:
                                     MockLogprob(logprob=-0.909731924533844,
                                                 rank=1,
                                                 decoded_token='ator'),
                                     1162:
                                     MockLogprob(logprob=-0.8929234743118286,
                                                 rank=2,
                                                 decoded_token=' year')
                                 }],
                                 finish_reason=None,
                                 stop_reason=None)
        ],
        finished=False),
    MockRequestOutput(
        request_id="test_request_id",
        prompt="I am a",
        prompt_token_ids=[1, 315, 837, 264],
        prompt_logprobs=[
            None, {
                315: MockLogprob(logprob=-4.37, rank=8, decoded_token='I'),
                422: MockLogprob(logprob=-1.25, rank=1, decoded_token='#')
            }, {
                837: MockLogprob(logprob=-2.62, rank=3, decoded_token='am'),
            }, {
                264: MockLogprob(logprob=-1.73, rank=1, decoded_token='a')
            }
        ],
        outputs=[
            MockCompletionOutput(index=1,
                                 text=' member of the',
                                 token_ids=[4292, 302,
                                            272],
                                 cumulative_logprob=-4.815703457221389,
                                 logprobs=[{
                                     4292:
                                     MockLogprob(logprob=-4.2740092277526855,
                                                 rank=4,
                                                 decoded_token=' member'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new'),
                                 }, {
                                     302:
                                     MockLogprob(logprob=-0.03010374866425991,
                                                 rank=1,
                                                 decoded_token=' of'),
                                     235290:
                                     MockLogprob(logprob=-2.2026185989379883,
                                                 rank=1,
                                                 decoded_token='-'),
                                     578:
                                     MockLogprob(logprob=-2.2026185989379883,
                                                 rank=2,
                                                 decoded_token=' and')
                                 }, {
                                     272:
                                     MockLogprob(logprob=-0.5115904808044434,
                                                 rank=1,
                                                 decoded_token=' the'),
                                     169181:
                                     MockLogprob(logprob=-8.463325500488281,
                                                 rank=196,
                                                 decoded_token=' aviator'),
                                     194366:
                                     MockLogprob(logprob=-2.463325023651123,
                                                 rank=1,
                                                 decoded_token=' Realtor')
                                 }],
                                 finish_reason='length',
                                 stop_reason=None),
            MockCompletionOutput(index=0,
                                 text=' consolidated or',
                                 token_ids=[22968, 601, 442],
                                 cumulative_logprob=-20.4010648727417,
                                 logprobs=[{
                                     22968:
                                     MockLogprob(logprob=-12.117759704589844,
                                                 rank=5308,
                                                 decoded_token=' consolid'),
                                     2032:
                                     MockLogprob(logprob=-3.0240092277526855,
                                                 rank=1,
                                                 decoded_token=' big'),
                                     17372:
                                     MockLogprob(logprob=-13.409988403320312,
                                                 rank=10489,
                                                 decoded_token=' crown'),
                                     888:
                                     MockLogprob(logprob=-4.4099884033203125,
                                                 rank=3,
                                                 decoded_token=' new'),
                                 }, {
                                     601:
                                     MockLogprob(logprob=-1.2847318649291992,
                                                 rank=2,
                                                 decoded_token='ated'),
                                     1028:
                                     MockLogprob(logprob=-0.909731924533844,
                                                 rank=1,
                                                 decoded_token='ator'),
                                     1162:
                                     MockLogprob(logprob=-0.8929234743118286,
                                                 rank=2,
                                                 decoded_token=' year')
                                 }, {
                                     442:
                                     MockLogprob(logprob=-6.998573303222656,
                                                 rank=188,
                                                 decoded_token=' or'),
                                     28725:
                                     MockLogprob(logprob=-3.7798233032226562,
                                                 rank=1,
                                                 decoded_token=','),
                                     1622:
                                     MockLogprob(logprob=-4.463325023651123,
                                                 rank=2,
                                                 decoded_token=' New'),
                                     576:
                                     MockLogprob(logprob=-4.463325023651123,
                                                 rank=3,
                                                 decoded_token=' of')
                                 }],
                                 finish_reason='length',
                                 stop_reason=None)
        ],
        finished=True)
]


def _compare_tokens(expected_token, actual_token):
    return expected_token.id == actual_token.id and expected_token.text == actual_token.text and \
           expected_token.special_token == actual_token.special_token and \
           expected_token.log_prob == actual_token.log_prob


class TestVllmUtils(unittest.TestCase):

    def setUp(self):
        sys.modules['vllm'] = MagicMock()
        sys.modules['vllm.outputs'] = MagicMock()
        sys.modules['vllm.lora.request'] = MagicMock()

    @mock.patch(
        'djl_python.rolling_batch.rolling_batch_vllm_utils.vLLMRequestOutput',
        new=MockRequestOutput)
    def test_multiple_sequences(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        parameters = {
            "max_new_tokens": 3,
            "details": True,
            "decoder_input_details": True,
            "best_of": 2,
            "n": 2,
            "top_n_tokens": 3
        }

        # 1. Creates the request
        req = Request(0,
                      "I am a",
                      parameters=parameters.copy(),
                      output_formatter=_json_output_formatter,
                      tokenizer=tokenizer)
        self.assertEqual(
            TextGenerationOutput(request_id=0,
                                 input=TextInput(
                                     request_id=0,
                                     input_text="I am a",
                                     parameters=parameters,
                                     output_formatter=_json_output_formatter,
                                     tokenizer=tokenizer),
                                 sequences={},
                                 finished=False), req.request_output)

        # 2. Creates the request cache
        mock_request_cache = OrderedDict(
            {"test_request_id": {
                "request_output": req.request_output
            }})

        # Test update_request_cache_with_output
        for vllm_request_output in example_request_output:
            djl_python.rolling_batch.rolling_batch_vllm_utils.update_request_cache_with_output(
                mock_request_cache, vllm_request_output, tokenizer)
        expected_sequences = {
            0:
            Sequence(tokens=[
                Token(id=22968,
                      text=' consolid',
                      log_prob=-12.117759704589844,
                      special_token=None),
                Token(id=601,
                      text='ated',
                      log_prob=-1.2847318649291992,
                      special_token=None),
                Token(id=442,
                      text=' or',
                      log_prob=-6.998573303222656,
                      special_token=None)
            ],
                     top_tokens=[[
                         Token(id=22968,
                               text=' consolid',
                               log_prob=-12.117759704589844,
                               special_token=None),
                         Token(id=2032,
                               text=' big',
                               log_prob=-3.0240092277526855,
                               special_token=None),
                         Token(id=17372,
                               text=' crown',
                               log_prob=-13.409988403320312,
                               special_token=None),
                         Token(id=888,
                               text=' new',
                               log_prob=-4.4099884033203125,
                               special_token=None)
                     ],
                                 [
                                     Token(id=601,
                                           text='ated',
                                           log_prob=-1.2847318649291992,
                                           special_token=None),
                                     Token(id=1028,
                                           text='ator',
                                           log_prob=-0.909731924533844,
                                           special_token=None),
                                     Token(id=1162,
                                           text=' year',
                                           log_prob=-0.8929234743118286,
                                           special_token=None)
                                 ],
                                 [
                                     Token(id=442,
                                           text=' or',
                                           log_prob=-6.998573303222656,
                                           special_token=None),
                                     Token(id=28725,
                                           text=',',
                                           log_prob=-3.7798233032226562,
                                           special_token=None),
                                     Token(id=1622,
                                           text=' New',
                                           log_prob=-4.463325023651123,
                                           special_token=None),
                                     Token(id=576,
                                           text=' of',
                                           log_prob=-4.463325023651123,
                                           special_token=None)
                                 ]],
                     finish_reason='length',
                     cumulative_log_prob=-20.4010648727417,
                     stop_reason=None),
            1:
            Sequence(
                tokens=[
                    Token(id=4292,
                          text=' member',
                          log_prob=-4.2740092277526855,
                          special_token=None),
                    Token(id=302,
                          text=' of',
                          log_prob=-0.03010374866425991,
                          special_token=None),
                    Token(id=272,
                          text=' the',
                          log_prob=-0.5115904808044434,
                          special_token=None)
                ],
                top_tokens=[[
                    Token(id=4292,
                          text=' member',
                          log_prob=-4.2740092277526855,
                          special_token=None),
                    Token(id=2032,
                          text=' big',
                          log_prob=-3.0240092277526855,
                          special_token=None),
                    Token(id=888,
                          text=' new',
                          log_prob=-4.4099884033203125,
                          special_token=None)
                ],
                            [
                                Token(id=302,
                                      text=' of',
                                      log_prob=-0.03010374866425991,
                                      special_token=None),
                                Token(id=235290,
                                      text='-',
                                      log_prob=-2.2026185989379883,
                                      special_token=None),
                                Token(id=578,
                                      text=' and',
                                      log_prob=-2.2026185989379883,
                                      special_token=None)
                            ],
                            [
                                Token(id=272,
                                      text=' the',
                                      log_prob=-0.5115904808044434,
                                      special_token=None),
                                Token(id=169181,
                                      text=' aviator',
                                      log_prob=-8.463325500488281,
                                      special_token=None),
                                Token(id=194366,
                                      text=' Realtor',
                                      log_prob=-2.463325023651123,
                                      special_token=None)
                            ]],
                finish_reason='length',
                cumulative_log_prob=-4.815703457221389,
                stop_reason=None,
            )
        }

        self.assertEqual(len(expected_sequences),
                         len(req.request_output.sequences))
        for seq_index, sequence in expected_sequences.items():
            actual_sequence = req.request_output.sequences[seq_index]
            self.assertEqual(sequence.finish_reason,
                             actual_sequence.finish_reason)
            self.assertEqual(sequence.stop_reason, actual_sequence.stop_reason)
            self.assertEqual(sequence.cumulative_log_prob,
                             actual_sequence.cumulative_log_prob)
            self.assertEqual(len(sequence.tokens), len(actual_sequence.tokens))
            for token_index, token in enumerate(sequence.tokens):
                self.assertTrue(
                    _compare_tokens(token,
                                    actual_sequence.tokens[token_index]))
            for top_tokens_index, top_tokens in enumerate(sequence.top_tokens):
                self.assertEqual(
                    len(top_tokens),
                    len(actual_sequence.top_tokens[top_tokens_index]))
                for token_index, token in enumerate(top_tokens):
                    self.assertTrue(
                        _compare_tokens(
                            token, actual_sequence.top_tokens[top_tokens_index]
                            [token_index]))
