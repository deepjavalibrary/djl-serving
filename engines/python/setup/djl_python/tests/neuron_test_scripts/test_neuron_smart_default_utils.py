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

import json
import copy
import unittest
from unittest.mock import patch
from djl_python.neuron_utils.neuron_smart_default_utils import NeuronSmartDefaultUtils

MODEL_CONFIG_2B = {
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 22,
    "num_key_value_heads": 4,
    "vocab_size": 32000
}

MODEL_CONFIG_8B = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256
}

MODEL_CONFIG_70B = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "max_position_embeddings": 131072,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "vocab_size": 128256
}

MODEL_CONFIG_UNIT = {
    "hidden_size": 1,
    "intermediate_size": 1,
    "max_position_embeddings": 1,
    "model_type": "llama",
    "num_attention_heads": 1,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "vocab_size": 1
}


class TestNeuronSmartDefaultUtils(unittest.TestCase):

    def test_get_available_cores(self):
        with patch('subprocess.check_output') as mock_check_output:
            mock_check_output.return_value = b'[{"core": 0}, {"core": 1}, {"core": 2}, {"core": 3}]'
            assert NeuronSmartDefaultUtils.get_available_cores() == 4

    def test_get_available_cores_exception(self):
        with patch('subprocess.check_output') as mock_check_output:
            mock_check_output.side_effect = Exception(
                'Error getting available cores')
            assert NeuronSmartDefaultUtils.get_available_cores() == 0

    def test_get_llama_like_parameters_exception(self):
        model_config = {}
        assert NeuronSmartDefaultUtils.get_llama_like_parameters(
            model_config) == 0

    def test_apply_smart_defaults_2b_model(self):
        properties = {}
        model_config = copy.deepcopy(MODEL_CONFIG_2B)
        utils = NeuronSmartDefaultUtils(available_cores=4)
        utils.apply_smart_defaults(properties, model_config)
        assert properties["n_positions"] == 2048
        assert properties["tensor_parallel_degree"] == 1
        assert properties["max_rolling_batch_size"] == 32

    def test_apply_smart_defaults_8b_model(self):
        properties = {}
        model_config = copy.deepcopy(MODEL_CONFIG_8B)
        utils = NeuronSmartDefaultUtils(available_cores=2)
        utils.apply_smart_defaults(properties, model_config)
        assert properties["n_positions"] == 4096
        assert properties["tensor_parallel_degree"] == 2
        assert properties["max_rolling_batch_size"] == 16

    def test_apply_smart_defaults_70b_model(self):
        properties = {}
        model_config = copy.deepcopy(MODEL_CONFIG_70B)
        utils = NeuronSmartDefaultUtils(available_cores=32)
        utils.apply_smart_defaults(properties, model_config)
        assert properties["n_positions"] == 4096
        assert properties["tensor_parallel_degree"] == 32
        assert properties["max_rolling_batch_size"] == 32

    def test_apply_smart_defaults_unit_model(self):
        model_config = copy.deepcopy(MODEL_CONFIG_UNIT)
        utils = NeuronSmartDefaultUtils(available_cores=1)
        assert utils.get_model_parameters(model_config) == 12


if __name__ == '__main__':
    unittest.main()
