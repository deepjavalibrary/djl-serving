import json
import unittest
from djl_python.rolling_batch.rolling_batch import Request, Token, _json_output_formatter, _jsonlines_output_formatter, \
    RollingBatch


class TestRollingBatch(unittest.TestCase):

    def test_json_fmt(self):
        req = Request(0, "This is a wonderful day", {"max_new_tokens": 256})
        req.set_next_token(Token(244, "He", -0.334532), _json_output_formatter)
        print(req.get_next_token(), end='')
        assert req.get_next_token() == '{"generated_text": "He'
        req.set_next_token(Token(576, "llo", -0.123123),
                           _json_output_formatter)
        print(req.get_next_token(), end='')
        assert req.get_next_token() == 'llo'
        req.set_next_token(Token(4558, " world", -0.567854),
                           _json_output_formatter, True, 'length')
        print(req.get_next_token(), end='')
        assert req.get_next_token() == ' world"}'

    def test_jsonlines_fmt(self):
        req = Request(0, "This is a wonderful day", {"max_new_tokens": 256})
        req.set_next_token(Token(244, "He", -0.334532),
                           _jsonlines_output_formatter)
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": [244],
                "text": "He",
                "log_prob": -0.334532
            }
        }
        req.set_next_token(Token(576, "llo", -0.123123),
                           _jsonlines_output_formatter)
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": [576],
                "text": "llo",
                "log_prob": -0.123123
            }
        }
        req.set_next_token(Token(4558, " world", -0.567854),
                           _jsonlines_output_formatter, True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": [4558],
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "Hello world"
        }

    def test_return_full_text(self):
        req = Request(0, "This is a wonderful day", {
            "max_new_tokens": 256,
            "return_full_text": True,
        })

        final_str = []
        req.set_next_token(Token(244, "He", -0.334532), _json_output_formatter)
        final_str.append(req.get_next_token())
        req.set_next_token(Token(576, "llo", -0.123123),
                           _json_output_formatter)
        final_str.append(req.get_next_token())
        req.set_next_token(Token(4558, " world", -0.567854),
                           _json_output_formatter, True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json, end='')
        assert final_json == {
            "generated_text": "This is a wonderful dayHello world",
        }

        req = Request(0, "This is a wonderful day", {
            "max_new_tokens": 256,
            "return_full_text": True
        })
        req.set_next_token(Token(244, "He", -0.334532),
                           _jsonlines_output_formatter)
        req.set_next_token(Token(576, "llo", -0.123123),
                           _jsonlines_output_formatter)
        req.set_next_token(Token(4558, " world", -0.567854),
                           _jsonlines_output_formatter, True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": [4558],
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "This is a wonderful dayHello world",
        }

    def test_details(self):
        req = Request(0, "This is a wonderful day", {
            "max_new_tokens": 256,
            "details": True
        })
        final_str = []
        req.set_next_token(Token(244, "He", -0.334532), _json_output_formatter)
        final_str.append(req.get_next_token())
        req.set_next_token(Token(576, "llo", -0.123123),
                           _json_output_formatter)
        final_str.append(req.get_next_token())
        req.set_next_token(Token(4558, " world", -0.567854),
                           _json_output_formatter, True, 'length')
        final_str.append(req.get_next_token())
        final_json = json.loads(''.join(final_str))
        print(final_json)
        assert final_json == {
            "generated_text": "Hello world",
            "details": {
                "finish_reason":
                "length",
                "generated_tokens":
                3,
                "tokens": [{
                    "id": [244],
                    "text": "He",
                    "log_prob": -0.334532
                }, {
                    "id": [576],
                    "text": "llo",
                    "log_prob": -0.123123
                }, {
                    "id": [4558],
                    "text": " world",
                    "log_prob": -0.567854
                }]
            }
        }
        # Jsonlines tests
        req = Request(0, "This is a wonderful day", {
            "max_new_tokens": 256,
            "details": True
        })
        req.set_next_token(Token(244, "He", -0.334532),
                           _jsonlines_output_formatter)
        req.set_next_token(Token(576, "llo", -0.123123),
                           _jsonlines_output_formatter)
        req.set_next_token(Token(4558, " world", -0.567854),
                           _jsonlines_output_formatter, True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            "token": {
                "id": [4558],
                "text": " world",
                "log_prob": -0.567854
            },
            "generated_text": "Hello world",
            "details": {
                "finish_reason": "length",
                "generated_tokens": 3,
            }
        }

    def test_custom_fmt(self):

        def custom_fmt(token: Token, first_token: bool, last_token: bool,
                       details: dict, generated_tokens: str):
            result = {
                "token_id": token.id,
                "token_text": token.text,
                "request_id": token.request_id
            }
            if last_token:
                result["finish_reason"] = details["finish_reason"]
            return json.dumps(result) + "\n"

        class CustomRB(RollingBatch):

            def preprocess_requests(self, requests):
                pass

            def postprocess_results(self):
                pass

            def inference(self, input_data, parameters):
                pass

        rb = CustomRB(output_formatter=custom_fmt)

        req = Request(132, "This is a wonderful day", {
            "max_new_tokens": 256,
            "details": True
        })
        final_str = []
        req.set_next_token(Token(244, "He", -0.334532), rb.output_formatter)
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': [244],
            'token_text': 'He',
            "request_id": 132
        }
        req.set_next_token(Token(576, "llo", -0.123123), rb.output_formatter)
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': [576],
            'token_text': 'llo',
            "request_id": 132
        }
        req.set_next_token(Token(4558, " world", -0.567854),
                           rb.output_formatter, True, 'length')
        print(req.get_next_token(), end='')
        assert json.loads(req.get_next_token()) == {
            'token_id': [4558],
            'token_text': ' world',
            'finish_reason': 'length',
            "request_id": 132
        }


if __name__ == '__main__':
    unittest.main()
