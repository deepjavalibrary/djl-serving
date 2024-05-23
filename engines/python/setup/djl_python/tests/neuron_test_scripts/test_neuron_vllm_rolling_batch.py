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
import warnings
import json

import os
import sys

try:
    from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties
    from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
    from djl_python.tests.rolling_batch_test_scripts.generator import Generator
    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True

expected_text_30 = {
    "TinyLlama/TinyLlama-1.1B-Chat-v0.6": {
        1:
        "Hello, my name is [Your Name] and I am a [Your Job Title] at [Your Company Name]. I am interested in learning more about your company'",
        2:
        'The president of the United States is a man named Donald Trump.\n\n2. The president of the United States is a man named Donald Trump.\n\n3. The president',
        3:
        'The capital of France is Paris.\n\n2. The capital of the United States is Washington, D.C.\n\n3. The capital of Canada is Ott',
        4:
        "The future of AI is bright, and it's not just in the realm of science fiction. Artificial intelligence is already being used in a wide range of industries",
    }
}


@unittest.skipIf(SKIP_TEST, "Neuron dependencies are not available")
class TestNeuronVLLM(unittest.TestCase):

    def test_models(self):
        # === Preparation ===
        script_directory = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../../../"
        new_path = os.path.normpath(
            os.path.join(script_directory, relative_path))
        sys.path.append(new_path)

        # --- Models ---
        model_names = [
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
        ]

        # === Test ===
        for model_id in model_names:
            properties = {
                "tensor_parallel_degree": 1,
                "dtype": "fp16",
                "device": "neuron",
                "max_model_len": "128",
                "rolling_batch": "vllm",
                "max_rolling_batch_size": 4,
                "model_loading_timeout": 3600,
                "model_id": model_id
            }

            # ===================== neuron-vllm ============================
            rolling_batch = VLLMRollingBatch(model_id, properties)

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
            }.copy() for _ in range(len(input_str1))]

            gen.step(step=10, input_str_delta=input_str1, params_delta=params1)

            for _ in range(1):
                print('========== inference_1 ===========')
                input_str_delta = [
                    "Hello, my name is Hello, my name is Hello, my name is Hello, my name is",  # 21
                    "Hello, my name is Hello, my name is Hello, my name is"
                ]  # 16

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
            gen.step(step=200)
            accp_tkns = [[e for e in list_cnt if e > 0]
                         for list_cnt in gen.token_numbers.values()]
            for req_id, out in gen.output_all.items():
                out_dict = json.loads(''.join(out))
                out_str = out_dict["generated_text"]
                if req_id > min(4, len(input_str1)): continue
                print(
                    f"\n====req_id: {req_id}=====\n{gen.input_all[req_id][0] + out_str}\n"
                )
                if model_id in expected_text_30 and req_id in expected_text_30[
                        model_id]:
                    print(gen.input_all[req_id][0] + out_str)
                    expected_prefix_30_req_id = expected_text_30[model_id][
                        req_id]
                    backup_check = -req_id in expected_text_30[model_id] and (
                        gen.input_all[req_id][0] +
                        out_str)[:len(expected_text_30[model_id][-req_id]
                                      )] == expected_text_30[model_id][-req_id]
                    assert expected_prefix_30_req_id == (
                        gen.input_all[req_id][0] + out_str
                    )[:len(expected_prefix_30_req_id)] or backup_check
                elif req_id < 6:
                    warnings.warn(
                        f"\nWARNING:-----------v_v\nmodel_id = {model_id}, req_id = {req_id} is not asserted!\n\n",
                        UserWarning)

            # Reset
            rolling_batch.reset()
            rolling_batch.model = None
            rolling_batch = None
            import gc
            gc.collect()


if __name__ == '__main__':
    c = TestNeuronVLLM()
    c.test_models()
