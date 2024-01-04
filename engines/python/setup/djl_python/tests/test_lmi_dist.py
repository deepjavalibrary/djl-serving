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

import unittest

import torch
import os

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
from djl_python.tests.rolling_batch_test_scripts.generator import Generator, print_rank0


class TestLmiDist(unittest.TestCase):

    def test_models(self):
        model_names = [
            # "TheBloke/Llama-2-7B-Chat-fp16",
            # weight model.layers.0.self_attn.rotary_emb.inv_freq does not exist
            # "TheBloke/Llama-2-7B-Chat-AWQ",
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            # weight model.layers.0.self_attn.rotary_emb.inv_freq does not exist
            # "TinyLlama/TinyLlama-1.1B-python-v0.1",
            # g5.12xlarge single gpu ok. But no way to clear the gpu memory after running llama-2-7b thus cause OOM
            # "codellama/CodeLlama-7b-hf"
        ]
        expected_text_30 = {
            "TheBloke/Llama-2-7B-Chat-fp16": {
                1:
                'Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of experience. I am reaching out to you today',
                2:
                'The president of the United States is the head of the executive branch of the federal government and is one of the most powerful political figures in the world. The president is elected by the',
                3:
                'The capital of France is Paris. It is located in the northern central part of the country and is known for its stunning architecture, art museums, fashion, and',
                4:
                "The future of AI is bright, but it's not without its challenges. Here are some of the biggest challenges that AI will face in the future:",
                5:
                'Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is'
            },
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6": {
                1:
                "Hello, my name is [Your Name] and I am a [Your Job Title] at [Your Company Name]. I am interested in learning more about your company'",
                2:
                'The president of the United States is a man named Donald Trump.\n\n2. The president of the United States is a man named Donald Trump.\n\n3. The president',
                3:
                'The capital of France is Paris.\n\n2. The capital of the United States is Washington, D.C.\n\n3. The capital of Canada is Ott',
                4:
                "The future of AI is bright, and it's already here. With the help of AI, we can create more personalized experiences, automate repetitive tasks",
                5:
                'Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is'
            }
        }

        for model_id in model_names:
            properties = {
                "mpi_mode": "true",
                "tensor_parallel_degree": 1,
                "dtype": "fp16",
                "max_rolling_batch_size": 28,
                "model_loading_timeout": 3600,
                "paged_attention": "True",
                "model_id": model_id
            }

            # ===================== lmi_dist ============================
            device = int(os.environ.get("RANK", 0))
            properties["device"] = int(os.environ.get("RANK", 0))

            rolling_batch = LmiDistRollingBatch(model_id, device, properties)
            rolling_batch.output_formatter = None

            gen = Generator(rolling_batch=rolling_batch)

            print('========== init inference ===========')
            input_str1 = [
                "Hello, my name is",  # 6
                "The president of the United States is",  # 8
                "The capital of France is",  # 6
                "The future of AI is"
            ]  # 7

            params1 = [{
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.001
            }, {
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.001
            }, {
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.001
            }, {
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.001
            }]

            gen.step(step=10, input_str_delta=input_str1, params_delta=params1)

            for _ in range(1):
                print('========== inference_1 ===========')
                input_str_delta = [
                    "Hello, my name is Hello, my name is Hello, my name is Hello, my name is",  # 22
                    "Hello, my name is Hello, my name is Hello, my name is"
                ]  # 17

                params_delta = [{
                    "max_new_tokens": 100,
                    "do_sample": False,
                    "temperature": 0.001
                }, {
                    "max_new_tokens": 100,
                    "do_sample": False,
                    "temperature": 0.001
                }]

                gen.step(step=10,
                         input_str_delta=input_str_delta,
                         params_delta=params_delta)

            print('========== inference_infty ===========')
            gen.step(step=500)
            for req_id, out in gen.output_all.items():
                print_rank0(
                    f"\n====req_id: {req_id}=====\n{gen.input_all[req_id][0] + ''.join(out)}\n"
                )
                if model_id in expected_text_30 and req_id in expected_text_30:
                    assert expected_text_30[model_id][
                        req_id] == gen.input_all[req_id][0] + ''.join(out[:30])

            # Reset
            rolling_batch.reset()
            rolling_batch.model = None
            rolling_batch = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()


if __name__ == '__main__':
    unittest.main()
