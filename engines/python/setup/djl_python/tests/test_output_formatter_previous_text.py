import unittest
from unittest.mock import Mock

from djl_python.output_formatter import _jsonlines_chat_output_formatter
from djl_python.rolling_batch.rolling_batch_vllm_utils import update_multiple_sequences
from djl_python.request_io import (TextGenerationOutput, TextInput, Sequence,
                                   Token)


class _RecordingReasoningParser:
    """Captures the streaming triple handed to the parser."""

    def __init__(self):
        self.calls = []
        self.id_calls = []

    def extract_reasoning_content_streaming(self, previous_text, current_text,
                                            delta_text, previous_token_ids,
                                            current_token_ids,
                                            delta_token_ids):
        self.calls.append((previous_text, current_text, delta_text))
        self.id_calls.append(
            (previous_token_ids, current_token_ids, delta_token_ids))
        return Mock(model_dump=lambda exclude_unset: {"content": delta_text})


class TestPreviousTextSlicing(unittest.TestCase):
    """previous_text must be current_text minus the new token's text.

    A token can carry empty text: the detokenizer holds bytes until a multi-byte
    character completes, and a step that retires several ids attributes the text
    to the last of them. current_text[0:-len("")] is current_text[0:0], i.e. "",
    which would hand the parser a previous_text that contradicts current_text.
    """

    @staticmethod
    def _emit(tokens):
        parser = _RecordingReasoningParser()
        request_input = TextInput(request_id=0,
                                  input_text="",
                                  parameters={"reasoning_parser": parser})
        seq = Sequence()
        out = TextGenerationOutput(request_id=0, input=request_input)
        out.sequences = {0: seq}
        out.best_sequence_index = 0
        # Tokens arrive one at a time, and the formatter runs after each arrival.
        for i, (tid, text) in enumerate(tokens):
            seq.set_next_token(Token(tid, text, -0.5), i == len(tokens) - 1)
            _jsonlines_chat_output_formatter(out)
        return parser.calls

    def test_non_empty_token_text(self):
        calls = self._emit([(8508, "客"), (50292, "价")])
        self.assertEqual(("", "客", "客"), calls[0])
        self.assertEqual(("客", "客价", "价"), calls[1])

    def test_empty_token_text_keeps_previous_text_consistent(self):
        # Second token carries no text; previous_text must stay the full text so
        # far, not collapse to "".
        calls = self._emit([(8508, "客单"), (43720, "")])
        self.assertEqual(("", "客单", "客单"), calls[0])
        self.assertEqual(("客单", "客单", ""), calls[1])
        for previous_text, current_text, delta_text in calls:
            self.assertEqual(current_text, previous_text + delta_text)

    def test_first_token_empty(self):
        calls = self._emit([(8508, ""), (43720, "客单")])
        self.assertEqual(("", "", ""), calls[0])
        self.assertEqual(("", "客单", "客单"), calls[1])


class _Logprob:

    def __init__(self, logprob, decoded_token):
        self.logprob = logprob
        self.rank = 1
        self.decoded_token = decoded_token


class _CompletionOutput:

    def __init__(self, token_ids, text, isolated, finish_reason=None):
        self.index = 0
        self.token_ids = list(token_ids)
        self.text = text
        self.finish_reason = finish_reason
        self.cumulative_logprob = -1.0
        self.logprobs = [{
            tid: _Logprob(-0.5, iso)
        } for tid, iso in zip(token_ids, isolated)]


class TestMultiTokenStepStreaming(unittest.TestCase):
    """The rolling batch appends a whole engine step, then the formatter drains the
    step one token at a time. Anything derived from the sequence must therefore be
    bounded by the iterator position, not by the full token list.
    """

    @staticmethod
    def _run(steps):
        parser = _RecordingReasoningParser()
        request_input = TextInput(request_id=0,
                                  input_text="",
                                  parameters={"reasoning_parser": parser})
        out = TextGenerationOutput(request_id=0, input=request_input)
        out.best_sequence_index = 0
        for i, (token_ids, delta, isolated) in enumerate(steps):
            completion = _CompletionOutput(
                token_ids,
                delta,
                isolated,
                finish_reason="length" if i == len(steps) - 1 else None)
            vllm_output = Mock(kv_transfer_params=None)
            vllm_output.outputs = [completion]
            update_multiple_sequences(out, vllm_output)
            seq = out.sequences[0]
            while seq.has_next_token():
                _jsonlines_chat_output_formatter(out)
        return parser

    def test_multi_token_step_keeps_the_streaming_triple_consistent(self):
        # Step two retires three ids for one delta, which is what speculative
        # decoding produces.
        parser = self._run([
            ([15496], "Hello", ["Hello"]),
            ([8508, 43720, 50292], "客单价", ["\ufffd", "\ufffd单", "价"]),
        ])
        calls = parser.calls
        self.assertEqual(4, len(calls))
        for previous_text, current_text, delta_text in calls:
            self.assertEqual(current_text, previous_text + delta_text)
        lengths = [len(previous_text) for previous_text, _, _ in calls]
        self.assertEqual(sorted(lengths), lengths,
                         "previous_text went backwards")
        self.assertEqual("Hello客单价", calls[-1][1])
        # The token-id triple must track the drain position, not the whole
        # sequence: current_token_ids is ids[:index+1] and previous is one shorter.
        all_ids = [15496, 8508, 43720, 50292]
        for index, (prev_ids, cur_ids,
                    delta_ids) in enumerate(parser.id_calls):
            self.assertEqual(all_ids[:index + 1], cur_ids)
            self.assertEqual(all_ids[:index], prev_ids)
            self.assertEqual([all_ids[index]], delta_ids)

    def test_multi_token_step_keeps_delimiter_delta_for_parser(self):
        # A reasoning delimiter that shares a multi-token step with a split
        # character must reach the parser with its own delta, not "": if it were
        # blanked, the parser would see the delimiter id with empty text and let
        # "</think>" leak into content.
        parser = self._run([
            ([1001, 2002,
              3003], "</think>Answer 客", ["</think>", "Answer", "\ufffd"]),
        ])
        deltas = [delta_text for _, _, delta_text in parser.calls]
        self.assertEqual(["</think>", "Answer", " 客"], deltas)
        for previous_text, current_text, delta_text in parser.calls:
            self.assertEqual(current_text, previous_text + delta_text)


class _RecordingToolParser:
    """Captures the streaming triple handed to a tool parser."""

    def __init__(self):
        self.calls = []
        self.id_calls = []

    def extract_tool_calls_streaming(self, previous_text, current_text,
                                     delta_text, previous_token_ids,
                                     current_token_ids, delta_token_ids,
                                     request):
        self.calls.append((previous_text, current_text, delta_text))
        self.id_calls.append(
            (previous_token_ids, current_token_ids, delta_token_ids))
        return None


class TestToolParserPreviousText(unittest.TestCase):
    """The tool-calling branch derives the same streaming triple as the reasoning
    branch and must stay consistent when a token carries no text."""

    @staticmethod
    def _emit(tokens):
        parser = _RecordingToolParser()
        chat_params = Mock(tools=[{"type": "function"}], tool_choice="auto")
        request_input = TextInput(request_id=0,
                                  input_text="",
                                  parameters={
                                      "chat_params": chat_params,
                                      "tool_parser": parser,
                                  })
        seq = Sequence()
        out = TextGenerationOutput(request_id=0, input=request_input)
        out.sequences = {0: seq}
        out.best_sequence_index = 0
        for i, (tid, text) in enumerate(tokens):
            seq.set_next_token(Token(tid, text, -0.5), i == len(tokens) - 1)
            _jsonlines_chat_output_formatter(out)
        return parser.calls

    def test_empty_token_text_keeps_previous_text_consistent(self):
        calls = self._emit([(8508, "客单"), (43720, "")])
        self.assertEqual(("", "客单", "客单"), calls[0])
        self.assertEqual(("客单", "客单", ""), calls[1])
        for previous_text, current_text, delta_text in calls:
            self.assertEqual(current_text, previous_text + delta_text)


if __name__ == '__main__':
    unittest.main()
