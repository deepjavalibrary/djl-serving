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

try:
    from djl_python.transformers_neuronx import TransformersNeuronXService
    from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties, TnXGenerationStrategy
    from djl_python.neuron_utils.model_loader import TNXModelLoader, OptimumModelLoader
    from djl_python.rolling_batch.neuron_rolling_batch import NeuronRollingBatch
    from djl_python.neuron_utils.utils import build_context_length_estimates
    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True


@parameterized
@unittest.skipIf(SKIP_TEST, "Neuron dependencies are not available")
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
        return patch.object(TNXModelLoader,
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
            expected.rolling_batch_strategy = TnXGenerationStrategy.naive_rolling_batch
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
            self.service.set_rolling_batch(test_properties)

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
        "model_loader": "tnx"
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

    @parameters([{
        "initial_value": 512,
        "smart_default": 512
    }, {
        "initial_value": 8192,
        "smart_default": 4096
    }])
    def test_smart_defaults(self, params):
        # Setup
        self.default_properties.pop('n_positions')
        test_properties = self.default_properties
        self.service.config = self.config_builder(test_properties)
        self.service.model_config = AutoConfig.from_pretrained(
            test_properties['model_id'])
        self.service.model_config.max_position_embeddings = params[
            'initial_value']

        # Test
        self.service.set_max_position_embeddings()

        # Evaluate
        self.assertEqual(self.service.config.n_positions,
                         params['smart_default'])

    def tearDown(self):
        del self.service
        del self.default_properties


@parameterized
@unittest.skipIf(SKIP_TEST, "Neuron dependencies are not available")
class TestTransformersNeuronXUtils(unittest.TestCase):

    @parameters([{
        "initial_value": 64,
        "smart_default": [64]
    }, {
        "initial_value": 512,
        "smart_default": [128, 512]
    }, {
        "initial_value": 8192,
        "smart_default": [128, 1024, 2048, 4096, 8192]
    }])
    def test_build_context_length_estimate(self, params):
        # Setup
        # Test
        output = build_context_length_estimates(params['initial_value'])

        # Evaluate
        self.assertListEqual(output, params['smart_default'])


if __name__ == '__main__':
    unittest.main()
