import unittest

from djl_python.huggingface import filter_unsupported_generate_kwargs


class TestHuggingFace(unittest.TestCase):

    def test_filter_unsupported_generate_kwargs_removes_chat_only_params(self):
        parameters = {
            "temperature": 0.6,
            "max_new_tokens": 4096,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "ignore_eos": True,
        }
        filtered = filter_unsupported_generate_kwargs(parameters)
        self.assertEqual(filtered, {
            "temperature": 0.6,
            "max_new_tokens": 4096,
        })

    def test_filter_unsupported_generate_kwargs_keeps_supported_params(self):
        parameters = {
            "temperature": 1.0,
            "top_p": 0.9,
            "stop": ["</s>"],
            "seed": 42,
        }
        filtered = filter_unsupported_generate_kwargs(parameters)
        self.assertEqual(filtered, parameters)

    def test_filter_unsupported_generate_kwargs_empty_input(self):
        self.assertEqual(filter_unsupported_generate_kwargs({}), {})


if __name__ == "__main__":
    unittest.main()
