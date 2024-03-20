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

import unittest
from transformers import AutoConfig, AutoTokenizer
from unittest.mock import MagicMock, Mock, patch
from djl_python.tests.utils import parameterized, parameters, mock_import_modules

MOCK_MODULES = [
    "torch_neuronx", "neuronxcc", "transformers_neuronx",
    "transformers_neuronx.config", "transformers_neuronx.module",
    "transformers_neuronx.gpt2", "transformers_neuronx.gpt2.model", "optimum",
    "optimum.neuron", "optimum.neuron.utils", "optimum.neuron.generation",
    "optimum.neuron.utils.version_utils", "optimum.exporters",
    "optimum.exporters.neuron", "optimum.exporters.neuron.model_configs",
    "djl_python.transformers_neuronx_scheduler.optimum_modeling",
    "optimum.exporters.tasks", "diffusers", "diffusers.models",
    "diffusers.models.unet_2d_condition",
    "diffusers.models.attention_processor"
]
mock_import_modules(MOCK_MODULES)

from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties
from djl_python.transformers_neuronx import TransformersNeuronXService
from djl_python.neuron_utils.model_loader import TNXModelLoader, OptimumModelLoader
from djl_python.rolling_batch.neuron_rolling_batch import NeuronRollingBatch


@parameterized
class TestTransformerNeuronXService(unittest.TestCase):

    def setUp(self):
        self.service = TransformersNeuronXService()
        self.default_properties = dict({
            "model_id": "hf-internal-testing/tiny-random-gpt2",
            "tensor_parallel_degree": 1,
            "batch_size": 1,
            "dtype": "fp16",
            "n_positions": 128,
        })

    @staticmethod
    def config_builder(properties):
        return TransformerNeuronXProperties(**properties)

    @staticmethod
    def patch_auto_tokenizer(params):
        mock_response = Mock()
        mock_response.pad_token_id = params.get('pad_token_id', None)
        mock_response.eos_token_id = "eos_token_id"
        return patch.object(AutoTokenizer,
                            "from_pretrained",
                            return_value=mock_response)

    @staticmethod
    def patch_load_model():
        return patch.object(TNXModelLoader,
                            "load_model",
                            return_value="mock_model")

    @staticmethod
    def patch_model_loader():
        return patch.object(TNXModelLoader, "__init__", return_value=None)

    @staticmethod
    def patch_partition():
        return patch.object(OptimumModelLoader,
                            "partition",
                            return_value="mock_model")

    @staticmethod
    def patch_neuron_rolling_batch():
        return patch.object(NeuronRollingBatch, "__init__", return_value=None)

    @parameters([{
        "task": "feature-extraction"
    }, {
        "context_length_estimate": "32, 64"
    }, {
        "load_in_8bit": True
    }, {
        "rolling_batch": "auto",
        "batch_size": 1,
        "max_rolling_batch_size": 4
    }, {
        "model_id":
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    }, {
        "revision": "91c0fe31d692dd8448d9bc06e8d1877345009e3b"
    }])
    def test_set_configs(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}
        expected = self.config_builder(test_properties)
        if expected.task is None:
            expected.task = "text-generation"
        if expected.rolling_batch != "disable":
            expected.batch_size = expected.max_rolling_batch_size
        model_loader_classes = [
            TNXModelLoader.__class__.__name__,
            OptimumModelLoader.__class__.__name__
        ]

        # Test
        self.service.set_configs(test_properties)

        # Evaluate
        self.assertDictEqual(self.service.config.__dict__, expected.__dict__)
        self.assertIn(self.service._model_loader_class.__class__.__name__,
                      model_loader_classes)

    @parameters([{
        "trust_remote_code": True
    }, {
        "pad_token_id": "pad_token_id"
    }, {
        "revision": "mock_revision"
    }])
    def test_set_tokenizer(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}
        self.service.config = self.config_builder(test_properties)

        # Test
        with self.patch_auto_tokenizer(params) as mock_tokenizer:
            self.service.set_tokenizer()

        # Evaluate
        mock_tokenizer.assert_called_once_with(
            test_properties['model_id'],
            trust_remote_code=test_properties.get('trust_remote_code', False),
            revision=test_properties.get('revision', None),
            padding_side="left")
        self.assertEqual(self.service.tokenizer.pad_token_id,
                         test_properties.get('pad_token_id', 'eos_token_id'))

    @parameters([{
        "rolling_batch": "disable"
    }, {
        "rolling_batch": "auto",
        "output_formatter": "jsonlines"
    }, {
        "rolling_batch": "mock"
    }])
    def test_set_rolling_batch(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}
        self.service.config = self.config_builder(test_properties)
        expected = True if params['rolling_batch'] != 'disable' else False

        # Test
        with self.patch_neuron_rolling_batch() as mock_rolling_batch:
            self.service.set_rolling_batch()

        # Evaluate
        self.assertEqual(mock_rolling_batch.called, expected)

    def test_set_model_loader(self):
        # Setup
        test_properties = self.default_properties
        self.service.config = self.config_builder(test_properties)
        self.service.model_config = {"test": "mock_model_config"}
        self.service._model_loader_class = MagicMock()

        # Test
        self.service.set_model_loader()

        # Evaluate
        self.service._model_loader_class.assert_called_once_with(
            config=self.service.config, model_config=self.service.model_config)

    @parameters([{
        "rolling_batch": "auto",
        "task": "text-generation",
        "max_rolling_batch_size": 4,
        "load_split_model": True
    }])
    def test_initialize(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}

        # Test
        with self.patch_model_loader() as mock_model_loader_class:
            with self.patch_load_model() as mock_model_loader:
                self.service.initialize(test_properties)

        # Evaluate
        mock_model_loader.assert_called()
        mock_model_loader_class.assert_called()
        self.assertTrue(self.service.initialized)

    @parameters([{
        "save_mp_checkpoint_path": "mock_path",
        "task": "text-generation"
    }])
    def test_partition(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}

        # Test
        with self.patch_partition() as mock_model_loader:
            self.service.partition(test_properties)

        # Evaluate
        mock_model_loader.assert_called()
        self.assertEqual(test_properties["save_mp_checkpoint_path"],
                         self.service.config.save_mp_checkpoint_path)
        self.assertTrue(self.service.initialized)

    def tearDown(self):
        del self.service
        del self.default_properties


if __name__ == '__main__':
    unittest.main()
