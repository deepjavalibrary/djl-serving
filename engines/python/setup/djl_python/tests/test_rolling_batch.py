import json
import unittest

from djl_python.request import Request
from djl_python.output_formatter import _json_output_formatter, _jsonlines_output_formatter, \
    _jsonlines_chat_output_formatter, _json_chat_output_formatter
from djl_python.request_io import Token, TextGenerationOutput, TextInput


class TestRollingBatch(unittest.TestCase):

    def test_json_fmt(self):
        req_input1 = TextInput(
            request_id=0,
            input_text="This is a wonderful day",
            parameters={"max_new_tokens": 256},
            output_formatter=_json_output_formatter,
        )
        req1 = Request(req_input1)
        req_input2 = TextInput(
            request_id=1,
            input_text="This is a wonderful day",
            parameters={
                "max_new_tokens": 256,
                "stream": False
            },
        )
        req2 = Request(req_input2)
        for req in [req1, req2]:
            req.set_next_token(Token(244, "He", -0.334532))
            req.get_next_token()
            req.set_next_token(Token(576, "llo", -0.123123))
            req.get_next_token()
            req.set_next_token(Token(4558, " world", -0.567854), True,
                               'length')
            print(req.get_next_token(), end='')
            assert req.get_next_token() == json.dumps(
                {"generated_text": "Hello world"})

    def test_json_speculative_decoding(self):
        req_input = TextInput(
            request_id=0,
            input_text="This is a wonderful day",
            parameters={"max_new_tokens": 256},
            output_formatter=_json_output_formatter,
        )
        req = Request(req_input)
        req.request_output = TextGenerationOutput(request_id=0,
                                                  input=req_input)
        req.request_output.finished = True
        req.request_output.set_next_token(Token(244, "He", -0.334532))
        req.request_output.set_next_token(Token(576, "llo", -0.123123))
        req.request_output.set_next_token(Token(4558, " world", -0.567854,
                                                True),
                                          is_last_token=True,
                                          finish_reason='length')

        self.assertEqual(req.get_next_token(), "")
        self.assertEqual(req.get_next_token(), "")
        self.assertEqual(req.get_next_token(),
                         json.dumps({"generated_text": "Hello world"}))

    def test_json_fmt_with_appending(self):
        req_input1 = TextInput(request_id=0,
                               input_text="This is a wonderful day",
                               parameters={"max_new_tokens": 256},
                               output_formatter=_json_output_formatter)
        req1 = Request(req_input1)
        req_input2 = TextInput(
            request_id=1,
            input_text="This is a wonderful day",
            parameters={
                "max_new_tokens": 256,
                "stream": False
            },
        )
        req2 = Request(req_input2)
        for req in [req1, req2]:
            req.set_next_token(Token(244, "He", -0.334532))
            req.get_next_token()
            req.set_next_token(Token(576, "llo", -0.123123))
            req.get_next_token()
            req.set_next_token(Token(4558, " world", -0.567854), True,
                               'length')
            print(req.get_next_token(), end='')
            assert req.get_next_token() == json.dumps(
                {"generated_text": "Hello world"})

    def test_fmt_hf_compat(self):
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True,
                          "details": True
                      },
                      output_formatter=_json_output_formatter,
                      tgi_compat=True))

        req.set_next_token(Token(244, "He", -0.334532))
        req.get_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        req.get_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        final_json = json.loads(req.get_next_token())
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
        req1 = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={"max_new_tokens": 256},
                      output_formatter=_jsonlines_output_formatter))
        req2 = Request(
            TextInput(request_id=1,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "stream": True
                      }))
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

    def test_jsonlines_speculative_decoding(self):
        request_input = TextInput(request_id=0,
                                  input_text="This is a wonderful day",
                                  parameters={"max_new_tokens": 256},
                                  output_formatter=_jsonlines_output_formatter)
        req = Request(request_input=request_input)
        req.request_output = TextGenerationOutput(request_id=0,
                                                  input=request_input)
        req.request_output.finished = True
        req.request_output.set_next_token(Token(244, "He", -0.334532))
        print(req.get_next_token(), end='')
        self.assertEqual(
            {"token": {
                "id": 244,
                "text": "He",
                "log_prob": -0.334532
            }}, json.loads(req.get_next_token()))
        req.reset_next_token()
        req.request_output.set_next_token(Token(576, "llo", -0.123123))
        print(req.get_next_token(), end='')
        self.assertEqual(
            {"token": {
                "id": 576,
                "text": "llo",
                "log_prob": -0.123123
            }}, json.loads(req.get_next_token()))
        req.reset_next_token()
        req.request_output.set_next_token(Token(4558, " world", -0.567854),
                                          is_last_token=True,
                                          finish_reason='length')
        print(req.get_next_token(), end='')
        self.assertEqual(
            {
                "token": {
                    "id": 4558,
                    "text": " world",
                    "log_prob": -0.567854
                },
                "generated_text": "Hello world"
            }, json.loads(req.get_next_token()))

    def test_sse_fmt(self):
        request_input = TextInput(request_id=0,
                                  input_text="This is a wonderful day",
                                  parameters={"max_new_tokens": 256},
                                  output_formatter="sse")
        req = Request(request_input)
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
        req = Request(
            TextInput(request_id=1,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "stream": True,
                      },
                      tgi_compat=True))
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

    def test_3p_fmt(self):
        req = Request(
            TextInput(request_id=1,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 1024,
                          "details": True,
                          "decoder_input_details": True,
                          "return_full_text": True,
                      },
                      output_formatter="3p"))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
        req.set_next_token(Token(244, "He", -0.334532))
        req.get_next_token()
        req.set_next_token(Token(244, "llo", -0.123123))
        req.get_next_token()
        req.set_next_token(Token(4558, " world", -0.567854))
        req.get_next_token()
        req.set_next_token(Token(245, "", -1, True), True, "length")
        output = json.loads(req.get_next_token())
        print(req.get_next_token())
        assert output == {
            "body": {
                "generation": "This is a wonderful dayHello world",
                "prompt_token_count": 5,
                "generation_token_count": 4,
                "stop_reason": "length"
            },
            "content_type": "application/json",
            "metering": {
                "inputTokenCount": 5,
                "outputTokenCount": 4,
            },
        }

    def test_3p_stream_fmt(self):
        req = Request(
            TextInput(request_id=1,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 1024,
                          "details": True
                      },
                      output_formatter="3p_stream"))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
        req.set_next_token(Token(244, "He", -0.334532))
        next_token = json.loads(req.get_next_token())
        assert next_token == {
            "body": {
                "generation": "He",
                "prompt_token_count": 5,
                "generation_token_count": 1,
                "stop_reason": None,
            },
            "metering": {
                "inputTokenCount": 5,
                "outputTokenCount": 1,
            },
            "content_type": "application/jsonlines",
        }
        req.reset_next_token()
        req.set_next_token(Token(244, "llo", -0.123123))
        next_token = json.loads(req.get_next_token())
        assert next_token == {
            "body": {
                "generation": "llo",
                "prompt_token_count": None,
                "generation_token_count": 2,
                "stop_reason": None,
            },
            "metering": {
                "outputTokenCount": 2,
            },
            "content_type": "application/jsonlines",
        }
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854))
        next_token = json.loads(req.get_next_token())
        assert next_token == {
            "body": {
                "generation": " world",
                "prompt_token_count": None,
                "generation_token_count": 3,
                "stop_reason": None,
            },
            "metering": {
                "outputTokenCount": 3,
            },
            "content_type": "application/jsonlines",
        }
        req.reset_next_token()
        req.set_next_token(Token(-1, "", -1, True, "some error"), True,
                           "error")
        next_token = json.loads(req.get_next_token())
        assert next_token == {
            "body": {
                "generation": "",
                "prompt_token_count": None,
                "generation_token_count": 4,
                "stop_reason": "error",
            },
            "metering": {
                "outputTokenCount": 4,
            },
            "content_type": "application/jsonlines",
            "error": {
                "error_code": 400,
                "error_msg": "some error",
            }
        }

    def test_return_full_text(self):
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True,
                      },
                      output_formatter=_json_output_formatter))

        req.set_next_token(Token(244, "He", -0.334532))
        req.get_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        req.get_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')

        assert req.get_next_token() == json.dumps(
            {"generated_text": "This is a wonderful dayHello world"})

        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "return_full_text": True
                      },
                      output_formatter=_jsonlines_output_formatter))
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
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=_json_output_formatter))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
        req.set_next_token(Token(244, "He", -0.334532))
        req.get_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        req.get_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')

        final_json = json.loads(req.get_next_token())

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
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=_jsonlines_output_formatter))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
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
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True,
                          "decoder_input_details": True
                      },
                      output_formatter=_jsonlines_output_formatter))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", None),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
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
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
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
                    'id': 4,
                    'log_prob': None,
                    'text': 'This'
                }, {
                    'id': 5,
                    'log_prob': -0.4,
                    'text': 'is'
                }, {
                    'id': 6,
                    'log_prob': -0.5,
                    'text': 'a'
                }, {
                    'id': 7,
                    'log_prob': -0.6,
                    'text': 'wonderful'
                }, {
                    'id': 8,
                    'log_prob': -0.7,
                    'text': 'day'
                }],
            },
            "generated_text": "Hello world"
        }

    def test_chat_json(self):
        req = Request(
            TextInput(
                request_id=0,
                input_text="This is a wonderful day",
                parameters={
                    "max_new_tokens": 256,
                    "details": True,
                    "logprobs": True,
                },
                output_formatter=_json_chat_output_formatter,
            ))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
        req.set_next_token(Token(244, "He", -0.334532))
        req.get_next_token()
        req.reset_next_token()
        req.set_next_token(Token(576, "llo", -0.123123))
        req.get_next_token()
        req.reset_next_token()
        req.set_next_token(Token(4558, " world", -0.567854), True, 'length')
        output = json.loads(req.get_next_token())
        print(output)
        assert output['choices'] == [{
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
        assert output['usage'] == {
            'prompt_tokens': 5,
            'completion_tokens': 3,
            'total_tokens': 8,
        }

    def test_chat_jsonlines(self):
        req = Request(
            TextInput(request_id=0,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True,
                          "decoder_input_details": True,
                          "logprobs": True
                      },
                      output_formatter=_jsonlines_chat_output_formatter))
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

        req = Request(
            TextInput(request_id=132,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=custom_fmt))
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

        req = Request(
            TextInput(request_id=132,
                      input_text="This is a wonderful day",
                      parameters={
                          "max_new_tokens": 256,
                          "details": True
                      },
                      output_formatter=custom_fmt))
        req.request_output.prompt_tokens_details = [
            Token(4, "This", -0.3),
            Token(5, "is", -0.4),
            Token(6, "a", -0.5),
            Token(7, "wonderful", -0.6),
            Token(8, "day", -0.7),
        ]
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
            "prompt_tokens": 5
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
            5
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
            5
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

        req = Request(
            TextInput(request_id=132,
                      input_text="This is a wonderful day",
                      parameters=parameters,
                      output_formatter=custom_fmt_wait))
        print(req.request_input.parameters)
        assert req.request_input.parameters == parameters

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

        req = Request(
            TextInput(request_id=132,
                      input_text="This is a wonderful day",
                      parameters=parameters,
                      output_formatter=_json_output_formatter))
        print(parameters)
        assert parameters == {
            "max_new_tokens": 256,
            "details": True,
            "best_of": 2
        }

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

        req = Request(
            TextInput(request_id=132,
                      input_text="This is a wonderful day",
                      parameters=parameters,
                      output_formatter=_json_output_formatter))
        print(parameters)
        assert req.request_input.parameters == parameters

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
