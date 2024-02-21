#!/usr/bin/env python3
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
import os
import unittest
from djl_python.test_model import TestHandler
from djl_python import huggingface
from .rolling_batch.fake_rolling_batch import FakeRollingBatch


def override_rolling_batch(rolling_batch_type: str, is_mpi: bool,
                           model_config):
    return FakeRollingBatch


huggingface.get_rolling_batch_class_from_str = override_rolling_batch


class TestTestModel(unittest.TestCase):

    def test_all_code(self):
        model_id = "NousResearch/Nous-Hermes-Llama2-13b"
        handler = TestHandler(huggingface)
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "max_new_tokens": 256
            }
        }, {
            "inputs": "A little redhood is",
            "parameters": {
                "max_new_tokens": 50
            }
        }]
        serving_properties = {
            "engine": "Python",
            "rolling_batch": "auto",
            "model_id": model_id
        }
        result = handler.inference_rolling_batch(
            inputs, serving_properties=serving_properties)
        self.assertEqual(len(result), len(inputs))

    def test_with_env(self):
        envs = {
            "OPTION_MODEL_ID": "NousResearch/Nous-Hermes-Llama2-13b",
            "SERVING_LOAD_MODELS": "test::MPI=/opt/ml/model",
            "OPTION_ROLLING_BATCH": "auto"
        }
        for key, value in envs.items():
            os.environ[key] = value
        handler = TestHandler(huggingface)
        self.assertEqual(handler.serving_properties["model_id"],
                         envs["OPTION_MODEL_ID"])
        self.assertEqual(handler.serving_properties["rolling_batch"],
                         envs["OPTION_ROLLING_BATCH"])
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "max_new_tokens": 50
            }
        }, {
            "inputs": "A little redhood is",
            "parameters": {
                "min_new_tokens": 51,
                "max_new_tokens": 256
            }
        }]
        result = handler.inference_rolling_batch(inputs)
        self.assertEqual(len(result), len(inputs))
        self.assertTrue(len(result[1]) > len(result[0]))

        for key in envs.keys():
            os.environ[key] = ""
