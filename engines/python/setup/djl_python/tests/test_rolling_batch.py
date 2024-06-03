import json
import unittest

from djl_python.request import Request
from djl_python.output_formatter import _json_output_formatter, _jsonlines_output_formatter, \
    _jsonlines_chat_output_formatter, _json_chat_output_formatter
from djl_python.request_io import Token, RequestOutput, TextGenerationOutput


class TestRollingBatch(unittest.TestCase):

    def test_json_fmt(self):
        req1 = Request(0,
                       "This is a wonderful day",
                       parameters={"max_new_tokens": 256},
                       output_formatter=_json_output_formatter)
        req2 = Request(1,
                       "This is a wonderful day",
                       parameters={
                           "max_new_tokens": 256,
                           "stream": False
                       })
        for req in [req1, req2]:
            req.set_next_token(Token(244, "He", -0.334532))
            print(req.get_next_token(), end='')
            assert req.get_next_token() == '{"generated_text": "He'
            req.reset_next_token()
            req.set_next_token(Token(576, "llo", -0.123123))
            print(req.get_next_token(), end='')
            assert req.get_next_token() == 'llo'
            req.reset_next_token()
            req.set_next_token(Token(4558, " world", -0.567854), True,
                               'length')
            print(req.get_next_token(), end='')
            assert req.get_next_token() == ' world"}'
            req.reset_next_token()

    def test_json_fmt_with_appending(self):
        req1 = Request(0,
                       "This is a wonderful day",
                       parameters={"max_new_tokens": 256},
                       output_formatter=_json_output_formatter)
        req2 = Request(1,
                       "This is a wonderful day",
                       parameters={
                           "max_new_tokens": 256,
                           "stream": False
                       })
        for req in [req1, req2]:
            req.set_next_token(Token(244, "He", -0.334532))
            req.set_next_token(Token(576, "llo", -0.123123))
            print(req.get_next_token(), end='')
            assert req.get_next_token() == '{"generated_text": "Hello'
            req.reset_next_token()
            req.set_next_token(Token(4558, " world", -0.567854), True,
                               'length')
            print(req.get_next_token(), end='')
            assert req.get_next_token() == ' world"}'

    def test_fmt_hf_compat(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True,
                          "details": True
                      },
                      output_formatter=_json_output_formatter,
                      tgi_compat=True)

        final_str = []
        req.set_next_token(Token(244, "He", -0.334532))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json, end='')
        assert final_json == [{
            'generated_text': 'This is a wonderful dayHello world',
            'details': {
                'finish_reason':
                'length',
                'generated_tokens':
                3,
                'inputs':
                'This is a wonderful day',
                'tokens': [{
                    'id': 244,
                    'text': 'He',
                    'logprob': -0.334532
                }, {
                    'id': 576,
                    'text': 'llo',
                    'logprob': -0.123123
                }, {
                    'id': 4558,
                    'text': ' world',
                    'logprob': -0.567854
                }]
            }
        }]

    def test_jsonlines_fmt(self):
        req1 = Request(0,
                       "This is a wonderful day",
                       parameters={"max_new_tokens": 256},
                       output_formatter=_jsonlines_output_formatter)
        req2 = Request(1,
                       "This is a wonderful day",
                       parameters={
                           "max_new_tokens": 256,
                           "stream": True
                       })
        for req in [req1, req2]:
            req.set_next_token(Token(244, "He", -0.334532))
            print(req.get_next_token(), end='')
            assert json.loads(req.get_next_token()) == {
                "token": {
                    "id": 244,
                    "text": "He",
                    "log_prob": -0.334532
                }
            }
            req.reset_next_token()
            req.set_next_token(Token(576, "llo", -0.123123))
            print(req.get_next_token(), end='')
            assert json.loads(req.get_next_token()) == {
                "token": {
                    "id": 576,
                    "text": "llo",
                    "log_prob": -0.123123
                }
            }
            req.reset_next_token()
            req.set_next_token(Token(4558, " world", -0.567854), True,
                               'length')
            print(req.get_next_token(), end='')
            assert json.loads(req.get_next_token()) == {
                "token": {
                    "id": 4558,
                    "text": " world",
                    "log_prob": -0.567854
                },
                "generated_text": "Hello world"
            }

    def test_sse_fmt(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={"max_new_tokens": 256},
                      output_formatter="sse")
        req.set_next_token(Token(244, "He", -0.334532))
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 244,
                "text": "He",
                "log_prob": -0.334532
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 576,
                "text": "llo",
                "log_prob": -0.123123
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 4558,
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "Hello world"
        }

    def test_sse_tgi_compat_fmt(self):
        req = Request(1,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "stream": True
                      },
                      tgi_compat=True)
        req.set_next_token(Token(244, "He", -0.334532))
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 244,
                "text": "He",
                "logprob": -0.334532
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 576,
                "text": "llo",
                "logprob": -0.123123
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        next_token = req.get_next_token()
        print(next_token, end='')
        self.assertTrue(next_token.startswith("data: "))
        assert json.loads(next_token[5:]) == {
            "token": {
                "id": 4558,
                "text": " world",
                "logprob": -0.567854
            },
            "generated_text": "Hello world",
            "details": {
                "finish_reason": "length",
                "generated_tokens": 3,
                "inputs": "This is a wonderful day"
            }
        }

    def test_return_full_text(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True,
                      },
                      output_formatter=_json_output_formatter)

        final_str = []
        req.set_next_token(Token(244, "He", -0.334532))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json, end='')
        assert final_json == {
            "generated_text": "This is a wonderful dayHello world",
        }

        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True
                      },
                      output_formatter=_jsonlines_output_formatter)
        req.set_next_token(Token(244, "He", -0.334532))
        req.set_next_token(Token(576, "llo", -0.123123))
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token().splitlines()[-1]) == {
            "token": {
                "id": 4558,
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "This is a wonderful dayHello world",
        }

    def test_details(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=_json_output_formatter)
        final_str = []
        req.set_next_token(Token(244, "He", -0.334532))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json)
        assert final_json == {
            "generated_text": "Hello world",
            "details": {
                'inputs':
                'This is a wonderful day',
                "finish_reason":
                "length",
                "generated_tokens":
                3,
                "tokens": [{
                    "id": 244,
                    "text": "He",
                    "log_prob": -0.334532
                }, {
                    "id": 576,
                    "text": "llo",
                    "log_prob": -0.123123
                }, {
                    "id": 4558,
                    "text": " world",
                    "log_prob": -0.567854
                }]
            }
        }
        # Jsonlines tests
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=_jsonlines_output_formatter)
        req.set_next_token(Token(244, "He", -0.334532))
        req.set_next_token(Token(576, "llo", -0.123123))
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token().splitlines()[-1]) == {
            "token": {
                "id": 4558,
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "Hello world",
            "details": {
                'inputs': 'This is a wonderful day',
                "finish_reason": "length",
                "generated_tokens": 3,
            }
        }

    def test_details_jsonlines(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True,
                          "decoder_input_details": True
                      },
                      output_formatter=_jsonlines_output_formatter)
        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": 244,
                "text": "He",
                "log_prob": -0.334532
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": 576,
                "text": "llo",
                "log_prob": -0.123123
            }
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854),
                           True,
                           'length',
                           prompt_tokens_details=[
                               Token(id=123, text="This", log_prob=None),
                               Token(id=456, text="is", log_prob=0.456),
                               Token(id=789, text="a", log_prob=0.789),
                               Token(id=124, text="wonderful", log_prob=0.124),
                               Token(id=356, text="day", log_prob=0.356)
                           ])
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": 4558,
                "text": " world",
                "log_prob": -0.567854
            },
            'details': {
                'finish_reason':
                'length',
                'generated_tokens':
                3,
                'inputs':
                'This is a wonderful day',
                'prefill': [{
                    'id': 123,
                    'log_prob': None,
                    'text': 'This'
                }, {
                    'id': 456,
                    'log_prob': 0.456,
                    'text': 'is'
                }, {
                    'id': 789,
                    'log_prob': 0.789,
                    'text': 'a'
                }, {
                    'id': 124,
                    'log_prob': 0.124,
                    'text': 'wonderful'
                }, {
                    'id': 356,
                    'log_prob': 0.356,
                    'text': 'day'
                }],
            },
            "generated_text": "Hello world"
        }

    def test_chat_json(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True,
                          "logprobs": True
                      },
                      output_formatter=_json_chat_output_formatter)
        final_str = []
        req.set_next_token(Token(244, "He", -0.334532))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        final_str.append(req.get_next_token())
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json)
        assert final_json['choices'] == [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'Hello world'
            },
            'logprobs': {
                'content': [{
                    'token':
                    'He',
                    'logprob':
                    -0.334532,
                    'bytes': [72, 101],
                    'top_logprobs': [{
                        'token': 'He',
                        'logprob': -0.334532,
                        'bytes': [72, 101]
                    }]
                }, {
                    'token':
                    'llo',
                    'logprob':
                    -0.123123,
                    'bytes': [108, 108, 111],
                    'top_logprobs': [{
                        'token': 'llo',
                        'logprob': -0.123123,
                        'bytes': [108, 108, 111]
                    }]
                }, {
                    'token':
                    ' world',
                    'logprob':
                    -0.567854,
                    'bytes': [32, 119, 111, 114, 108, 100],
                    'top_logprobs': [{
                        'token': ' world',
                        'logprob': -0.567854,
                        'bytes': [32, 119, 111, 114, 108, 100]
                    }]
                }]
            },
            'finish_reason': 'length'
        }]
        assert final_json['usage'] == {
            'prompt_tokens': 0,
            'completion_tokens': 3,
            'total_tokens': 3
        }

    def test_chat_jsonlines(self):
        req = Request(0,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True,
                          "decoder_input_details": True,
                          "logprobs": True
                      },
                      output_formatter=_jsonlines_chat_output_formatter)
        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token())["choices"] == [{
            "index":
            0,
            "delta": {
                "content": "He",
                "role": "assistant"
            },
            "logprobs": [{
                "content": [{
                    "token":
                    "He",
                    "logprob":
                    -0.334532,
                    "bytes": [72, 101],
                    "top_logprobs": [{
                        "token": -0.334532,
                        "logprob": -0.334532,
                        "bytes": [72, 101]
                    }]
                }]
            }],
            "finish_reason":
            None
        }]
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token())["choices"] == [{
            "index":
            0,
            "delta": {
                "content": "llo"
            },
            "logprobs": [{
                "content": [{
                    "token":
                    "llo",
                    "logprob":
                    -0.123123,
                    "bytes": [108, 108, 111],
                    "top_logprobs": [{
                        "token": -0.123123,
                        "logprob": -0.123123,
                        "bytes": [108, 108, 111]
                    }]
                }]
            }],
            "finish_reason":
            None
        }]
        req.reset_next_token()

        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token())["choices"] == [{
            "index":
            0,
            "delta": {
                "content": " world"
            },
            "logprobs": [{
                "content": [{
                    "token":
                    " world",
                    "logprob":
                    -0.567854,
                    "bytes": [32, 119, 111, 114, 108, 100],
                    "top_logprobs": [{
                        "token": -0.567854,
                        "logprob": -0.567854,
                        "bytes": [32, 119, 111, 114, 108, 100]
                    }]
                }]
            }],
            "finish_reason":
            "length"
        }]
        req.reset_next_token()

    def test_custom_fmt(self):

        def custom_fmt(token: Token, first_token: bool, last_token: bool,
                       details: dict, generated_tokens: str, id: int):
            result = {
                "token_id": token.id,
                "token_text": token.text,
                "request_id": token.request_id
            }
            if last_token:
                result["finish_reason"] = details["finish_reason"]
            return json.dumps(result) + "\n"

        req = Request(132,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=custom_fmt)
        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': 244,
            'token_text': 'He',
            "request_id": 132
        }
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': 576,
            'token_text': 'llo',
            "request_id": 132
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': 4558,
            'token_text': ' world',
            'finish_reason': 'length',
            "request_id": 132
        }

    def test_custom_fmt_with_detailed_data_retrival(self):

        def custom_fmt(token: Token, first_token: bool, last_token: bool,
                       details: dict, generated_tokens: str, id: int):
            result = details
            return json.dumps(result) + "\n"

        req = Request(132,
                      "This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      input_ids=[101, 1188, 1110, 170, 7310, 1285, 102],
                      output_formatter=custom_fmt)
        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'finish_reason': None,
            'generated_tokens': 1,
            'inputs': 'This is a wonderful day',
            'tokens': [{
                'id': 244,
                'log_prob': -0.334532,
                'text': 'He'
            }],
            "parameters": {
                "max_new_tokens": 256,
                "details": True
            },
            "prompt_tokens": 7
        }
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'finish_reason':
            None,
            'generated_tokens':
            2,
            'inputs':
            'This is a wonderful day',
            'tokens': [
                {
                    'id': 244,
                    'text': 'He',
                    'log_prob': -0.334532,
                },
                {
                    'id': 576,
                    'text': 'llo',
                    'log_prob': -0.123123,
                },
            ],
            "parameters": {
                "max_new_tokens": 256,
                "details": True
            },
            "prompt_tokens":
            7
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'finish_reason':
            'length',
            'generated_tokens':
            3,
            'inputs':
            'This is a wonderful day',
            'tokens': [{
                'id': 244,
                'text': 'He',
                'log_prob': -0.334532,
            }, {
                'id': 576,
                'text': 'llo',
                'log_prob': -0.123123,
            }, {
                'id': 4558,
                'text': ' world',
                'log_prob': -0.567854,
            }],
            "parameters": {
                "max_new_tokens": 256,
                "details": True
            },
            "prompt_tokens":
            7
        }

    def test_custom_fmt_wait_till_last(self):
        """ Test with custom formatter.
            Waits till last token is generated, and send out the whole response at once.
        """

        def custom_fmt_wait(request_output: TextGenerationOutput):
            sequence_index = request_output.best_sequence_index
            best_sequence = request_output.sequences[
                request_output.best_sequence_index]
            _, _, last_token = best_sequence.get_next_token()
            if last_token:
                tokens = best_sequence.tokens
                generated_text = ""
                for token in tokens:
                    generated_text += token.text
                result = {"generated_text": generated_text}
                parameters = request_output.input.parameters
                if parameters.get("details", False):
                    result["finish_reason"] = best_sequence.finish_reason
                    result["tokens"] = request_output.get_tokens_as_dict(
                        sequence_index)
                    result["generated_tokens"] = len(best_sequence.tokens)
                    result["inputs"] = request_output.input.input_text
                    result["parameters"] = parameters
                # Special handling for error case
                elif best_sequence.finish_reason == "error":
                    result["finish_reason"] = best_sequence.finish_reason
                return json.dumps(result) + "\n"
            return json.dumps("") + "\n"

        parameters = {"max_new_tokens": 256, "details": True, "stream": False}

        req = Request(132,
                      "This is a wonderful day",
                      parameters=parameters,
                      output_formatter=custom_fmt_wait)
        print(parameters)
        assert parameters == {"max_new_tokens": 256}

        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == ""
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == ""
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "generated_text":
            "Hello world",
            "finish_reason":
            "length",
            "tokens": [{
                "id": 244,
                "text": "He",
                "log_prob": -0.334532
            }, {
                "id": 576,
                "text": "llo",
                "log_prob": -0.123123
            }, {
                "id": 4558,
                "text": " world",
                "log_prob": -0.567854
            }],
            "generated_tokens":
            3,
            "inputs":
            "This is a wonderful day",
            "parameters": {
                "max_new_tokens": 256,
                "details": True,
                "stream": False
            },
        }

    def test_best_of(self):
        """ Test with custom formatter.
            Waits till last token is generated, and send out the whole response at once.
        """

        parameters = {"max_new_tokens": 256, "details": True, "best_of": 2}

        req = Request(132,
                      "This is a wonderful day",
                      parameters=parameters,
                      output_formatter=_json_output_formatter)
        print(parameters)
        assert parameters == {"max_new_tokens": 256, "best_of": 2}

        req.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        assert req.get_next_token() == ""
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        assert req.get_next_token() == ""
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "generated_text": "Hello world",
            "details": {
                "finish_reason":
                "length",
                "tokens": [{
                    "id": 244,
                    "text": "He",
                    "log_prob": -0.334532
                }, {
                    "id": 576,
                    "text": "llo",
                    "log_prob": -0.123123
                }, {
                    "id": 4558,
                    "text": " world",
                    "log_prob": -0.567854
                }],
                "generated_tokens":
                3,
                "inputs":
                "This is a wonderful day"
            }
        }

    def test_multiple_sequences(self):
        """ Test with multiple sequences.
        """

        parameters = {"max_new_tokens": 4, "details": True, "best_of": 2}

        req = Request(132,
                      "This is a wonderful day",
                      parameters=parameters,
                      output_formatter=_json_output_formatter)
        print(parameters)
        assert parameters == {"max_new_tokens": 4, "best_of": 2}

        req.request_output.set_next_token(Token(244, "He", -0.334532),
                                          sequence_index=0)
        req.request_output.set_next_token(Token(12, "I ", -0.563752),
                                          sequence_index=1)
        print(req.get_next_token(), end='')
        assert req.get_next_token() == ""
        req.reset_next_token()

        req.request_output.set_next_token(Token(576, "llo", -0.123123),
                                          sequence_index=0)
        req.request_output.set_next_token(Token(123, "am", -0.452051),
                                          sequence_index=1)

        print(req.get_next_token(), end='')
        assert req.get_next_token() == ""
        req.reset_next_token()
        req.request_output.set_next_token(Token(4558, " world", -0.567854),
                                          sequence_index=0)
        # best_sequence finishes before other sequences.
        req.request_output.set_next_token(Token(8763, " AI", -0.836312),
                                          sequence_index=1,
                                          is_last_token=True,
                                          finish_reason='eos_token')

        assert req.get_next_token() == ""
        req.reset_next_token()
        req.request_output.set_next_token(Token(9872, " programming",
                                                -0.835241),
                                          sequence_index=0,
                                          is_last_token=True,
                                          finish_reason='length')

        req.request_output.best_sequence_index = 1
        req.request_output.other_sequences_indices = [0]
        req.request_output.finished = True
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "generated_text": "I am AI",
            "details": {
                "finish_reason":
                "eos_token",
                "tokens": [{
                    "id": 12,
                    "text": "I ",
                    "log_prob": -0.563752
                }, {
                    "id": 123,
                    "text": "am",
                    "log_prob": -0.452051
                }, {
                    "id": 8763,
                    "text": " AI",
                    "log_prob": -0.836312
                }],
                "generated_tokens":
                3,
                "inputs":
                "This is a wonderful day",
                "best_of_sequences": [{
                    "finish_reason":
                    "length",
                    "generated_tokens":
                    4,
                    "tokens": [{
                        "id": 244,
                        "text": "He",
                        "log_prob": -0.334532
                    }, {
                        "id": 576,
                        "text": "llo",
                        "log_prob": -0.123123
                    }, {
                        "id": 4558,
                        "text": " world",
                        "log_prob": -0.567854
                    }, {
                        "id": 9872,
                        "text": " programming",
                        "log_prob": -0.835241
                    }],
                    "generated_text":
                    "Hello world programming",
                }]
            }
        }


if __name__ == '__main__':
    unittest.main()
