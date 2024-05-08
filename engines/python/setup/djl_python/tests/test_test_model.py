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
import json
import os
import unittest
from djl_python.test_model import TestHandler
from djl_python import huggingface
from .rolling_batch.fake_rolling_batch import FakeRollingBatch, FakeRollingBatchWithException


def override_rolling_batch(rolling_batch_type: str, is_mpi: bool,
                           model_config):
    return FakeRollingBatch


def override_rolling_batch_with_exception(rolling_batch_type: str,
                                          is_mpi: bool, model_config):
    return FakeRollingBatchWithException


class TestTestModel(unittest.TestCase):

    def test_all_code(self):
        model_id = "NousResearch/Nous-Hermes-Llama2-13b"
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch
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
        self.assertTrue(json.loads(result[0]), dict)
        self.assertTrue(json.loads(result[1]), dict)

    def test_with_env(self):
        envs = {
            "OPTION_MODEL_ID": "NousResearch/Nous-Hermes-Llama2-13b",
            "SERVING_LOAD_MODELS": "test::MPI=/opt/ml/model",
            "OPTION_ROLLING_BATCH": "auto",
            "OPTION_TGI_COMPAT": "true"
        }
        for key, value in envs.items():
            os.environ[key] = value
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch
        handler = TestHandler(huggingface)
        self.assertEqual(handler.serving_properties["model_id"],
                         envs["OPTION_MODEL_ID"])
        self.assertEqual(handler.serving_properties["rolling_batch"],
                         envs["OPTION_ROLLING_BATCH"])
        self.assertEqual(handler.serving_properties["tgi_compat"],
                         envs["OPTION_TGI_COMPAT"])
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
        # TGI compat tests
        self.assertTrue(json.loads(result[0]), list)
        self.assertTrue(json.loads(result[1]), list)

        for key in envs.keys():
            del os.environ[key]

    def test_with_tgi_compat_env(self):
        envs = {
            "OPTION_MODEL_ID": "NousResearch/Nous-Hermes-Llama2-13b",
            "SERVING_LOAD_MODELS": "test::MPI=/opt/ml/model",
            "OPTION_ROLLING_BATCH": "auto",
            "OPTION_TGI_COMPAT": "true"
        }
        for key, value in envs.items():
            os.environ[key] = value
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch
        handler = TestHandler(huggingface)
        self.assertEqual(handler.serving_properties["model_id"],
                         envs["OPTION_MODEL_ID"])
        self.assertEqual(handler.serving_properties["rolling_batch"],
                         envs["OPTION_ROLLING_BATCH"])
        self.assertEqual(handler.serving_properties["tgi_compat"],
                         envs["OPTION_TGI_COMPAT"])
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "max_new_tokens": 50
            },
            "stream": True
        }]
        result = handler.inference_rolling_batch(inputs)
        self.assertEqual(len(result), len(inputs))
        # TGI compat tests
        sse_result = result[0].split("\n\n")[:-1]
        loaded_result = []
        for row in sse_result:
            loaded_result.append(json.loads(row[6:]))
        self.assertTrue("details" in loaded_result[-1], loaded_result[-1])

        for key in envs.keys():
            del os.environ[key]

    def test_all_code_chat(self):
        model_id = "TheBloke/Llama-2-7B-Chat-fp16"
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch
        handler = TestHandler(huggingface)
        inputs = [{
            "inputs":
            "<|system|>You are a helpful assistant.</s><|user|>What is deep learning?</s>",
            "parameters": {
                "max_new_tokens": 50
            }
        }, {
            "inputs":
            "<|system|>You are a friendly chatbot who always responds in the style of a pirate</s><|user|>How many helicopters can a human eat in one sitting?</s>",
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

    def test_with_env_chat(self):
        envs = {
            "OPTION_MODEL_ID": "TheBloke/Llama-2-7B-Chat-fp16",
            "SERVING_LOAD_MODELS": "test::MPI=/opt/ml/model",
            "OPTION_ROLLING_BATCH": "auto"
        }
        for key, value in envs.items():
            os.environ[key] = value
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch
        handler = TestHandler(huggingface)
        self.assertEqual(handler.serving_properties["model_id"],
                         envs["OPTION_MODEL_ID"])
        self.assertEqual(handler.serving_properties["rolling_batch"],
                         envs["OPTION_ROLLING_BATCH"])
        inputs = [{
            "inputs":
            "<|system|>You are a helpful assistant.</s><|user|>What is deep learning?</s>",
            "parameters": {
                "max_new_tokens": 50
            }
        }, {
            "inputs":
            "<|system|>You are a friendly chatbot who always responds in the style of a pirate</s><|user|>How many helicopters can a human eat in one sitting?</s>",
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

    def test_exception_handling(self):
        huggingface.get_rolling_batch_class_from_str = override_rolling_batch_with_exception
        model_id = "NousResearch/Nous-Hermes-Llama2-13b"
        handler = TestHandler(huggingface)
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "min_new_tokens": 100,
                "max_new_tokens": 256,
            }
        }, {
            "inputs": "Hello world",
            "parameters": {
                "min_new_tokens": 100,
                "max_new_tokens": 512,
            }
        }]
        serving_properties = {
            "engine": "Python",
            "rolling_batch": "auto",
            "model_id": model_id
        }
        result = handler.inference_rolling_batch(
            inputs, serving_properties=serving_properties)
        for key, value in result.items():
            final_dict = json.loads(value)
            self.assertEqual(final_dict["details"]["finish_reason"], 'error')
        # test streaming
        inputs = [{
            "inputs": "The winner of oscar this year is",
            "parameters": {
                "min_new_tokens": 100,
                "max_new_tokens": 256,
            },
            "stream": True,
        }, {
            "inputs": "Hello world",
            "parameters": {
                "min_new_tokens": 100,
                "max_new_tokens": 512,
            },
            "stream": True,
        }]
        result = handler.inference_rolling_batch(
            inputs, serving_properties=serving_properties)
        for _, value in result.items():
            final_dict = json.loads(value.splitlines()[-1])
            self.assertEqual(final_dict["details"]["finish_reason"], 'error')

    def test_multi_token_output(self):
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        from collections import defaultdict

        properties = {
            "mpi_mode": "true",
            "tensor_parallel_degree": 1,
            "dtype": "fp16",
            "max_rolling_batch_size": 28,
            "model_loading_timeout": 3600,
            "max_rolling_batch_prefill_tokens": 2048,
            "paged_attention": "True"
        }

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        properties['spec_length'] = 5

        # ===================== lmi ============================
        device = int(os.environ.get("RANK", 0))

        properties["model_id"] = model_id
        properties["speculative_draft_model"] = draft_model_id
        properties["device"] = device
        rolling_batch = LmiDistRollingBatch(model_id, properties)

        print('========== init inference ===========')
        input_str1 = [
            "Hello, my name is",  # 6
            "The president of the United States is",  # 8
            "The capital of France is",  # 6
            "The future of AI is"
        ]

        params1 = [{
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.001,
            "logprobs": 5,
        }.copy() for _ in range(len(input_str1))]

        # gen.step(step=3, input_str_delta=input_str1, params_delta=params1)
        rolling_batch.inference(input_str1, params1)
        cum_logprob_cache = defaultdict()
        for request_id in rolling_batch.request_cache.keys():
            cum_logprob = rolling_batch.request_cache[request_id][
                'cumulative_logprob']
            logprobs = rolling_batch.request_cache[request_id]['logprobs']
            token_ids = rolling_batch.request_cache[request_id]['token_ids']
            assert len(logprobs) == len(token_ids) == 1
            assert cum_logprob == logprobs[0]
            cum_logprob_cache[request_id] = cum_logprob

        for _ in range(3):
            rolling_batch.inference(input_str1, params1)
            for request_id in rolling_batch.request_cache.keys():
                cum_logprob = rolling_batch.request_cache[request_id][
                    'cumulative_logprob']
                logprobs = rolling_batch.request_cache[request_id]['logprobs']
                token_ids = rolling_batch.request_cache[request_id][
                    'token_ids']
                assert len(logprobs) == len(
                    token_ids) == properties['spec_length']
                assert cum_logprob == cum_logprob_cache[request_id] + sum(
                    logprobs)
                cum_logprob_cache[request_id] = cum_logprob
