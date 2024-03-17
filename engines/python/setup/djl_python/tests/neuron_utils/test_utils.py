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

import copy
import unittest
from transformers import AutoTokenizer
from djl_python.tests.utils import parameterized, parameters, mock_import_modules

MOCK_MODULES = [
    "torch_neuronx",
    "neuronxcc",
    "transformers_neuronx",
    "djl_python.transformers_neuronx_scheduler.optimum_modeling",
    "optimum.neuron.generation",
    "optimum.neuron.utils",
    "optimum.neuron.utils.version_utils",
    "optimum.neuron",
    "optimum.modeling_base",
    "optimum.exporters",
    "optimum.exporters.tasks",
    "optimum.exporters.neuron",
    "optimum.exporters.neuron.model_configs",
    "optimum",
]
mock_import_modules(MOCK_MODULES)

from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties
from djl_python.neuron_utils.utils import parse_input_to_default_schema, parse_input_as_chat_completion

test_tokenizer_id = "TheBloke/Llama-2-13B-Chat-fp16"
simple_chat_completion = {
    "model":
    "test_model",
    "messages": [{
        "role": "system",
        "content": "You are a helpful assistant"
    }, {
        "role": "user",
        "content": "Hello!"
    }]
}
simple_input_chat_completion = {
    "inputs": [{
        "role": "system",
        "content": "You are a helpful assistant"
    }, {
        "role": "user",
        "content": "Hello!"
    }]
}
input_parameters = {
    "temperature": 0.7,
    "top_p": 0.1,
    "max_tokens": 256,
    "logprobs": True
}
complex_chat_completion = {**simple_chat_completion, **input_parameters}


class TestNeuronInputParsingUtils(unittest.TestCase):

    def setUp(self):
        self.default_properties = dict({
            "model_id": "hf-internal-testing/tiny-random-gpt2",
            "tensor_parallel_degree": 1,
            "batch_size": 1,
            "n_positions": 128,
        })
        self.config = TransformerNeuronXProperties(**self.default_properties)
        self.tokenizer = AutoTokenizer.from_pretrained(test_tokenizer_id)
        self.output_parameters = {
            "temperature": 0.7,
            "top_p": 0.1,
            "max_new_tokens": 256,
            "details": True
        }

    @staticmethod
    def assert_object_keys_and_values_match(obj1, obj2):
        for key in obj1:
            assert key in obj2, f"{key} is not in {obj2}"
            assert obj1[key] == obj2[
                key], f"{key}: {obj1[key]} does not equal {obj2[key]}"

    def test_simple_chat_completion_input(self):
        chat_completion_req = copy.deepcopy(simple_chat_completion)
        new_input_map = parse_input_to_default_schema(chat_completion_req,
                                                      self.tokenizer,
                                                      self.config)
        expected_output = [
            self.tokenizer.apply_chat_template(chat_completion_req["messages"],
                                               tokenize=False)
        ]
        assert "inputs" in new_input_map.keys()
        assert new_input_map[
            "inputs"] == expected_output, f"Output does not match: {new_input_map['inputs']}, {expected_output}"

    def test_chat_completion_input_with_params(self):
        chat_completion_req = copy.deepcopy(complex_chat_completion)
        new_input_map = parse_input_to_default_schema(chat_completion_req,
                                                      self.tokenizer,
                                                      self.config)
        expected_output = [
            self.tokenizer.apply_chat_template(chat_completion_req["messages"],
                                               tokenize=False)
        ]
        assert "inputs" in new_input_map.keys()
        assert new_input_map[
            "inputs"] == expected_output, f"Output does not match: {new_input_map['inputs']}, {expected_output}"
        self.assert_object_keys_and_values_match(new_input_map["parameters"],
                                                 self.output_parameters)

    def test_fail_chat_completion_no_tokenizer_support(self):
        chat_completion_req = copy.deepcopy(simple_chat_completion)
        with self.assertRaises(AttributeError):
            parse_input_to_default_schema(chat_completion_req, dict(),
                                          self.config)

    def test_simple_input_chat_completion_input(self):
        chat_completion_req = copy.deepcopy(simple_input_chat_completion)
        new_input_map = parse_input_as_chat_completion(chat_completion_req,
                                                       self.tokenizer,
                                                       self.config)
        expected_output = [
            self.tokenizer.apply_chat_template(chat_completion_req["inputs"],
                                               tokenize=False)
        ]
        assert "inputs" in new_input_map.keys()
        assert new_input_map[
            "inputs"] == expected_output, f"Output does not match: {new_input_map['inputs']}, {expected_output}"

    def test_input_chat_completion_input_with_params(self):
        chat_completion_req = copy.deepcopy({
            **simple_input_chat_completion, "parameters":
            self.output_parameters
        })
        new_input_map = parse_input_as_chat_completion(chat_completion_req,
                                                       self.tokenizer,
                                                       self.config)
        expected_output = [
            self.tokenizer.apply_chat_template(chat_completion_req["inputs"],
                                               tokenize=False)
        ]
        assert "inputs" in new_input_map.keys()
        assert new_input_map[
            "inputs"] == expected_output, f"Output does not match: {new_input_map['inputs']}, {expected_output}"
        self.assert_object_keys_and_values_match(new_input_map["parameters"],
                                                 self.output_parameters)

    def tearDown(self):
        del self.tokenizer
        del self.config
        del self.default_properties


if __name__ == '__main__':
    unittest.main()
