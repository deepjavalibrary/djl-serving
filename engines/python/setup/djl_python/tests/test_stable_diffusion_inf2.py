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
from unittest.mock import patch
from djl_python.tests.utils import parameterized, parameters, mock_import_modules

MOCK_MODULES = [
    "torch_neuronx", "transformers_neuronx", "transformers_neuronx.config",
    "transformers_neuronx.module", "optimum", "optimum.neuron", "diffusers",
    "diffusers.models", "diffusers.models.unet_2d_condition",
    "diffusers.models.attention_processor"
]
mock_import_modules(MOCK_MODULES)

from djl_python.properties_manager.sd_inf2_properties import StableDiffusionNeuronXProperties
from djl_python.stable_diffusion_inf2 import StableDiffusionNeuronXService
from djl_python.neuron_utils.model_loader import OptimumStableDiffusionLoader


@parameterized
class TestStableDiffusionNeuronXService(unittest.TestCase):

    def setUp(self):
        self.service = StableDiffusionNeuronXService()
        self.default_properties = dict({
            "model_id": "optimum/tiny-stable-diffusion-neuronx",
            "tensor_parallel_degree": 1,
            "batch_size": 2,
            "dtype": "bf16",
            "height": 64,
            "width": 64,
        })

    @staticmethod
    def config_builder(properties):
        return StableDiffusionNeuronXProperties(**properties)

    @staticmethod
    def patch_load_pipeline():
        return patch.object(OptimumStableDiffusionLoader,
                            "load_pipeline",
                            return_value="mock_pipeline")

    @staticmethod
    def patch_partition():
        return patch.object(OptimumStableDiffusionLoader,
                            "partition",
                            return_value="mock_pipeline")

    @parameters([{"use_auth_token": "mock-auth-token"}, {}])
    def test_get_pipeline_kwargs(self, params):
        # Setup
        test_properties = {**self.default_properties, **params}
        self.service.config = self.config_builder(test_properties)
        expected = {"torch_dtype": self.service.config.dtype}
        if 'use_auth_token' in params:
            expected['use_auth_token'] = self.service.config.use_auth_token

        # Test
        result = self.service.get_pipeline_kwargs()

        # Evaluate
        self.assertDictEqual(expected, result)
        self.assertEqual(self.service.pipeline_loader.__class__.__name__,
                         'OptimumStableDiffusionLoader')

    def test_initialize(self):
        # Setup
        test_properties = {**self.default_properties}

        # Test
        with self.patch_load_pipeline() as mock_pipeline_loader:
            self.service.initialize(test_properties)

        # Evaluate
        mock_pipeline_loader.assert_called()
        self.assertTrue(self.service.initialized)

    @parameters([{"save_mp_checkpoint_path": "mock_path"}])
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
