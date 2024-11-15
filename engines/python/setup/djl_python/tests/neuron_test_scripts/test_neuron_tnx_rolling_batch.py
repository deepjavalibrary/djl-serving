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
import json
import os

try:
    import transformers_neuronx
    from djl_python.transformers_neuronx import TransformersNeuronXService
    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True

expected_text_30 = {
    "TinyLlama/TinyLlama-1.1B-Chat-v0.6": {
        0:
        "Hello, my name is [Your Name] and I am a [Your Job Title] at [Your Company Name]. I am interested in learning more about your company'",
        1:
        'The president of the United States is a man named Donald Trump.\n\n2. The president of the United States is a man named Donald Trump.\n\n3. The president',
        2:
        'The capital of France is Paris.\n\n2. The capital of the United States is Washington, D.C.\n\n3. The capital of Canada is Ott',
        3:
        "The future of AI is bright, and it's already here. With the help of AI, we can create more personalized experiences, automate repetitive tasks, and even predict the future.",
    }
}


@unittest.skipIf(SKIP_TEST, "Neuron dependencies are not available")
class TestNeuronRollingBatch(unittest.TestCase):

    def test_models(self):
        # === Preparation ===
        from djl_python.tests.neuron_test_scripts.neuron_rb_generator import NeuronRollingBatchGenerator, SimulationSchedule

        # --- Models ---
        model_names = [
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
        ]

        # === Test ===
        for model_id in model_names:
            properties = {
                "tensor_parallel_degree": 2,
                "n_positions": "128",
                "rolling_batch": "tnx",
                "max_rolling_batch_size": 4,
                "model_id": model_id
            }

            # ===================== neuron-tnx ============================
            gen = NeuronRollingBatchGenerator()
            gen.init_neuron_service(properties)

            print('========== init inference ===========')
            input_str = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            params = [{
                "max_new_tokens": 100,
                "do_sample": False,
            }.copy() for _ in range(len(input_str))]

            test_input = SimulationSchedule(prompts=input_str,
                                            params=params,
                                            reqs_to_prefill=[1, 2, 1],
                                            wait_steps=[1, 4, 5])

            gen.simulator(test_input)

            for i, out in enumerate(gen.responses):
                out_dict = json.loads(''.join(out))
                out_str = out_dict["generated_text"]
                test_generation = input_str[i] + " " + out_str
                print(f"\n====req_id: {i}=====\n{test_generation}\n")
                if model_id in expected_text_30 and i in expected_text_30[
                        model_id]:
                    expected_prefix_30_req_id = expected_text_30[model_id][i]
                    assert expected_prefix_30_req_id == test_generation[:len(
                        expected_prefix_30_req_id)]

            gen.reset()
            del gen
            import gc
            gc.collect()

    def test_tiny_models(self):
        # === Preparation ===
        from djl_python.tests.neuron_test_scripts.neuron_rb_generator import NeuronRollingBatchGenerator, SimulationSchedule
        from djl_python.tests.neuron_test_scripts.tiny_models import artifacts
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # --- Models ---
        model_name_vs_artifacts = {
            "llama": "s3://djl-llm/llama-tiny-4k/",
            "gpt2": "s3://djl-llm/gpt2-tiny-4k/",
            "gptneox": "s3://djl-llm/gpt-neox-tiny-4k/",
            "bloom": "s3://djl-llm/bloom-tiny-4k/",
        }

        # === Test ===
        for model_id in model_name_vs_artifacts.keys():
            properties = {
                "tensor_parallel_degree": 2,
                "n_positions": "128",
                "rolling_batch": "tnx",
                "max_rolling_batch_size": 4,
                "model_loading_timeout": 3600,
                "model_id": model_name_vs_artifacts[model_id]
            }

            # ===================== neuron-tnx ============================
            gen = NeuronRollingBatchGenerator()
            gen.init_neuron_service(properties)

            print('========== init inference ===========')
            input_str = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            params = [{
                "max_new_tokens": 100,
                "do_sample": False,
                "ignore_eos": True,
            }.copy() for _ in range(len(input_str))]

            test_input = SimulationSchedule(prompts=input_str,
                                            params=params,
                                            reqs_to_prefill=[1, 2, 1],
                                            wait_steps=[1, 4, 5])

            gen.simulator(test_input)
            gen.reset()
            del gen
            import gc
            gc.collect()


if __name__ == '__main__':
    unittest.main()
