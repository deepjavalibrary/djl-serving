#!/usr/bin/env python3

import os
import subprocess
import llm.prepare as prepare
import llm.client as client
import rb_client as rb_client

djl_version = os.environ.get('TEST_DJL_VERSION', '').strip()


class Runner:

    def __init__(self, container, test_name=None):
        self.container = container
        self.test_name = test_name

        # Compute flavor
        if djl_version is not None and len(djl_version) > 0:
            if container == "cpu":
                flavor = djl_version
            else:
                flavor = f"{djl_version}-{container}"
        else:
            flavor = f"{container}-nightly"

        self.image = f"deepjavalibrary/djl-serving:{flavor}"

    def __enter__(self):
        # os.system(f'docker pull {self.image}')
        os.system('rm -rf models')
        return self

    def __exit__(self, *args):
        if self.test_name is not None:
            esc_test_name = self.test_name.replace("/", "-")
            os.system(f"mkdir -p all_logs/{esc_test_name}")
            os.system(f"cp -r logs all_logs/{esc_test_name}")
        subprocess.run(["./remove_container.sh"], check=True)
        os.system("cat logs/serving.log")

    def launch(self, env_vars=None, cmd=None):
        if env_vars is not None:
            with open("docker_env", "w") as f:
                f.write(env_vars)
        else:
            if os.path.isfile("docker_env"):
                os.remove("docker_env")

        if cmd is None:
            cmd = 'serve -m test=file:/opt/ml/model/test/'

        model_dir = os.path.join(os.getcwd(), 'models')
        subprocess.run(
            f'./launch_container.sh {self.image} {model_dir} {self.container} {cmd}'
            .split(),
            check=True)


class TestHfHandler:
    # Runs on g5.12xl
    def test_gpt_neo(self):
        with Runner('lmi', 'test_gpt4all_lora') as r:
            prepare.build_hf_handler_model("gpt-neo-2.7b")
            r.launch()
            client.run("huggingface gpt-neo-2.7b".split())

    def test_bloom_7b(self):
        with Runner('lmi', 'bloom-7b1') as r:
            prepare.build_hf_handler_model("bloom-7b1")
            r.launch()
            client.run("huggingface bloom-7b1".split())

    def test_llama2_7b(self):
        with Runner('lmi', 'llama-2-7b') as r:
            prepare.build_hf_handler_model("llama-2-7b")
            r.launch()
            client.run("huggingface llama-2-7b".split())

    def test_gptj_6B(self):
        with Runner('lmi', 'gpt-j-6b') as r:
            prepare.build_hf_handler_model("gpt-j-6b")
            r.launch()
            client.run("huggingface gpt-j-6b".split())

    def test_gpt4all_lora(self):
        with Runner('lmi', 'gpt4all-lora') as r:
            prepare.build_hf_handler_model("gpt4all-lora")
            r.launch()
            client.run("huggingface gpt4all-lora".split())

    def test_streaming_bigscience_bloom_3b(self):
        with Runner('lmi', 'bigscience/bloom-3b') as r:
            prepare.build_hf_handler_model("bigscience/bloom-3b")
            r.launch("CUDA_VISIBLE_DEVICES=1,2")
            client.run("huggingface bigscience/bloom-3b".split())

    def test_streaming_t5_large(self):
        with Runner('lmi', 't5-large') as r:
            prepare.build_hf_handler_model("t5-large")
            r.launch("CUDA_VISIBLE_DEVICES=1")
            client.run("huggingface t5-large".split())


class TestTrtLlmHandler1:
    # Runs on g5.12xl
    def test_llama2_13b_tp4(self):
        with Runner('tensorrt-llm', 'llama2-13b') as r:
            prepare.build_trtllm_handler_model("llama2-13b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm llama2-13b".split())

    def test_falcon_triton(self):
        with Runner('tensorrt-llm', 'falcon-7b') as r:
            prepare.build_trtllm_handler_model("falcon-7b")
            r.launch("CUDA_VISIBLE_DEVICES=0")
            client.run("trtllm falcon-7b".split())

    def test_internlm_7b(self):
        with Runner('tensorrt-llm', 'internlm-7b') as r:
            prepare.build_trtllm_handler_model("internlm-7b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm internlm-7b".split())

    def test_baichuan2_13b(self):
        with Runner('tensorrt-llm', 'baichuan2-13b') as r:
            prepare.build_trtllm_handler_model("baichuan2-13b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm baichuan2-13b".split())

    def test_chatglm3_6b(self):
        with Runner('tensorrt-llm', 'chatglm3-6b') as r:
            prepare.build_trtllm_handler_model("chatglm3-6b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm chatglm3-6b".split())

    def test_gpt2(self):
        with Runner('tensorrt-llm', 'gpt2') as r:
            prepare.build_trtllm_handler_model("gpt2")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm gpt2".split())

    def test_santacoder(self):
        with Runner('tensorrt-llm', 'santacoder') as r:
            prepare.build_trtllm_handler_model("santacoder")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm santacoder".split())


class TestTrtLlmHandler2:
    # Runs on g5.12xl
    def test_llama2_7b_hf_smoothquant(self):
        with Runner('tensorrt-llm', 'llama2-7b-smoothquant') as r:
            prepare.build_trtllm_handler_model("llama2-7b-smoothquant")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm llama2-7b-smoothquant".split())

    def test_mistral(self):
        with Runner('tensorrt-llm', 'mistral-7b') as r:
            prepare.build_trtllm_handler_model("mistral-7b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm mistral-7b".split())

    def test_gpt_j_6b(self):
        with Runner('tensorrt-llm', 'gpt-j-6b') as r:
            prepare.build_trtllm_handler_model("gpt-j-6b")
            r.launch("CUDA_VISIBLE_DEVICES=0")
            client.run("trtllm gpt-j-6b".split())

    def test_qwen_7b(self):
        with Runner('tensorrt-llm', 'qwen-7b') as r:
            prepare.build_trtllm_handler_model("qwen-7b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm qwen-7b".split())


class TestSchedulerSingleGPU:
    # Runs on g5.12xl

    def test_gpt2(self):
        with Runner('lmi', 'gpt2') as r:
            prepare.build_rolling_batch_model("gpt2")
            r.launch()
            rb_client.run("correctness gpt2".split())

    def test_bllm(self):
        with Runner('lmi', 'bloom-560m') as r:
            prepare.build_rolling_batch_model("bloom-560m")
            r.launch()
            rb_client.run("scheduler_single_gpu bloom-560m".split())


class TestSchedulerMultiGPU:
    # Runs on g5.12xl

    def test_gptj_6b(self):
        with Runner('lmi', 'gpt-j-6b') as r:
            prepare.build_rolling_batch_model("gpt-j-6b")
            r.launch()
            rb_client.run("scheduler_multi_gpu gpt-j-6b".split())


class TestLmiDist1:
    # Runs on g5.12xl

    def test_gpt_neox_20b(self):
        with Runner('lmi', 'gpt-neox-20b') as r:
            prepare.build_lmi_dist_model("gpt-neox-20b")
            r.launch()
            client.run("lmi_dist gpt-neox-20b".split())

    def test_falcon_7b(self):
        with Runner('lmi', 'falcon-7b') as r:
            prepare.build_lmi_dist_model("falcon-7b")
            r.launch()
            client.run("lmi_dist falcon-7b".split())

    def test_falcon2_11b(self):
        with Runner('lmi', 'falcon-11b') as r:
            prepare.build_lmi_dist_model("falcon-11b")
            r.launch()
            client.run("lmi_dist falcon-11b".split())

    def test_gpt2(self):
        with Runner('lmi', 'gpt2') as r:
            envs = [
                "OPTION_MAX_ROLLING_BATCH_SIZE=2",
                "OPTION_OUTPUT_FORMATTER=jsonlines",
                "TENSOR_PARALLEL_DEGREE=1", "HF_MODEL_ID=gpt2",
                "OPTION_TASK=text-generation", "OPTION_ROLLING_BATCH=lmi-dist"
            ]
            r.launch("\n".join(envs))
            client.run("lmi_dist gpt2".split())

    def test_mpt_7b(self):
        with Runner('lmi', 'mpt-7b') as r:
            prepare.build_lmi_dist_model("mpt-7b")
            r.launch()
            client.run("lmi_dist mpt-7b".split())

    def test_llama2_tiny_autoawq(self):
        with Runner('lmi', 'llama-2-tiny-autoawq') as r:
            prepare.build_lmi_dist_model("llama-2-tiny")
            r.launch(
                "CUDA_VISIBLE_DEVICES=0,1,2,3",
                cmd=
                "partition --model-dir /opt/ml/input/data/training --save-mp-checkpoint-path /opt/ml/input/data/training/aot"
            )
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3",
                     cmd="serve -m test=file:/opt/ml/model/test/aot")
            client.run("lmi_dist llama-2-tiny".split())
        os.system('sudo rm -rf models')


class TestLmiDist2:
    # Runs on g5.12xl

    def test_gpt_neox_20b(self):
        with Runner('lmi', 'octocoder') as r:
            prepare.build_lmi_dist_model("octocoder")
            r.launch()
            client.run("lmi_dist octocoder".split())

    def test_speculative_llama_13b(self):
        with Runner('lmi', 'speculative-llama-13b') as r:
            prepare.build_lmi_dist_model("speculative-llama-13b")
            r.launch()
            client.run("lmi_dist speculative-llama-13b".split())

    def test_starcoder2_7b(self):
        with Runner('lmi', 'starcoder2-7b') as r:
            prepare.build_lmi_dist_model("starcoder2-7b")
            r.launch()
            client.run("lmi_dist starcoder2-7b".split())

    def test_gemma_2b(self):
        with Runner('lmi', 'gemma-2b') as r:
            prepare.build_lmi_dist_model("gemma-2b")
            r.launch()
            client.run("lmi_dist gemma-2b".split())

    def test_llama2_13b_gptq(self):
        with Runner('lmi', 'llama2-13b-gptq') as r:
            prepare.build_lmi_dist_model("llama2-13b-gptq")
            r.launch()
            client.run("lmi_dist llama2-13b-gptq".split())

    def test_mistral_7b(self):
        with Runner('lmi', 'mistral-7b') as r:
            prepare.build_lmi_dist_model("mistral-7b")
            r.launch()
            client.run("lmi_dist mistral-7b".split())

    def test_llama2_7b_32k(self):
        with Runner('lmi', 'llama2-7b-32k') as r:
            prepare.build_lmi_dist_model("llama2-7b-32k")
            r.launch()
            client.run("lmi_dist llama2-7b-32k".split())

    def test_mistral_7b_128k_awq(self):
        with Runner('lmi', 'mistral-7b-128k-awq') as r:
            prepare.build_lmi_dist_model("mistral-7b-128k-awq")
            r.launch()
            client.run("lmi_dist mistral-7b-128k-awq".split())

    def test_llama2_7b_chat(self):
        with Runner('lmi', 'llama2-7b-chat') as r:
            prepare.build_lmi_dist_model("llama2-7b-chat")
            r.launch()
            client.run("lmi_dist_chat llama2-7b-chat".split())


class TestVllm1:
    # Runs on g5.12xl

    def test_gpt_neox_20b(self):
        with Runner('lmi', 'gpt-neox-20b') as r:
            prepare.build_vllm_model("gpt-neox-20b")
            r.launch()
            client.run("vllm gpt-neox-20b".split())

    def test_mistral_7b(self):
        with Runner('lmi', 'mistral-7b') as r:
            prepare.build_vllm_model("mistral-7b")
            r.launch()
            client.run("vllm mistral-7b".split())

    def test_phi2(self):
        with Runner('lmi', 'phi-2') as r:
            prepare.build_vllm_model("phi-2")
            r.launch()
            client.run("vllm phi-2".split())

    def test_starcoder2_7b(self):
        with Runner('lmi', 'starcoder2-7b') as r:
            prepare.build_vllm_model("starcoder2-7b")
            r.launch()
            client.run("vllm starcoder2-7b".split())

    def test_gemma_2b(self):
        with Runner('lmi', 'gemma-2b') as r:
            prepare.build_vllm_model("gemma-2b")
            r.launch()
            client.run("vllm gemma-2b".split())

    def test_llama2_7b_chat(self):
        with Runner('lmi', 'llama2-7b-chat') as r:
            prepare.build_vllm_model("llama2-7b-chat")
            r.launch()
            client.run("vllm_chat llama2-7b-chat".split())


class TestVllmLora:
    # Runs on g5.12xl

    def test_lora_unmerged(self):
        with Runner('lmi', 'llama-7b-unmerged-lora') as r:
            prepare.build_vllm_model("llama-7b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama-7b-unmerged-lora".split())

    def test_lora_unmerged_overflow(self):
        with Runner('lmi', 'llama-7b-unmerged-lora-overflow') as r:
            prepare.build_vllm_model("llama-7b-unmerged-lora-overflow")
            r.launch()
            client.run("vllm_adapters llama-7b-unmerged-lora-overflow".split())

    def test_lora_awq_llama2_13b(self):
        with Runner('lmi', 'llama2-13b-awq-unmerged-lora') as r:
            prepare.build_vllm_model("llama2-13b-awq-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama2-13b-awq-unmerged-lora".split())

    def test_lora_mistral_7b(self):
        with Runner('lmi', 'mistral-7b-unmerged-lora') as r:
            prepare.build_vllm_model("mistral-7b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters mistral-7b-unmerged-lora".split())

    def test_lora_awq_mistral_7b(self):
        with Runner('lmi', 'mistral-7b-awq-unmerged-lora') as r:
            prepare.build_vllm_model("mistral-7b-awq-unmerged-lora")
            r.launch()
            client.run("vllm_adapters mistral-7b-awq-unmerged-lora".split())

    def test_lora_llama3_8b(self):
        with Runner('lmi', 'llama3-8b-unmerged-lora') as r:
            prepare.build_vllm_model("llama3-8b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama3-8b-unmerged-lora".split())


class TestLmiDistLora:
    # Runs on g5.12xl

    def test_lora_unmerged(self):
        with Runner('lmi', 'llama-7b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("llama-7b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters llama-7b-unmerged-lora".split())

    def test_lora_unmerged_overflow(self):
        with Runner('lmi', 'llama-7b-unmerged-lora-overflow') as r:
            prepare.build_lmi_dist_model("llama-7b-unmerged-lora-overflow")
            r.launch()
            client.run(
                "lmi_dist_adapters llama-7b-unmerged-lora-overflow".split())

    def test_lora_awq_llama2_13b(self):
        with Runner('lmi', 'llama2-13b-awq-unmerged-lora') as r:
            prepare.build_lmi_dist_model("llama2-13b-awq-unmerged-lora")
            r.launch()
            client.run(
                "lmi_dist_adapters llama2-13b-awq-unmerged-lora".split())

    def test_lora_mistral_7b(self):
        with Runner('lmi', 'mistral-7b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("mistral-7b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters mistral-7b-unmerged-lora".split())

    def test_lora_awq_mistral_7b(self):
        with Runner('lmi', 'mistral-7b-awq-unmerged-lora') as r:
            prepare.build_lmi_dist_model("mistral-7b-awq-unmerged-lora")
            r.launch()
            client.run(
                "lmi_dist_adapters mistral-7b-awq-unmerged-lora".split())

    def test_lora_llama3_8b(self):
        with Runner('lmi', 'llama3-8b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("llama3-8b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters llama3-8b-unmerged-lora".split())
