import unittest

from djl_python.input_parser import parse_input_with_formatter
from djl_python.test_model import create_concurrent_batch_request


class InputParserTest(unittest.TestCase):

    def test_input_parameters(self):
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "max_new_tokens": 50
            },
            "stream": False
        }, {
            "inputs": "A little redhood is",
            "parameters": {
                "min_new_tokens": 51,
                "max_new_tokens": 256,
            },
            "stream": True
        }]

        serving_properties = {"rolling_batch": "disable"}

        inputs = create_concurrent_batch_request(
            inputs, serving_properties=serving_properties)
        parsed_input = parse_input_with_formatter(inputs)
