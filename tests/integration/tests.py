#!/usr/bin/env python3

import os
import subprocess
import logging
import pytest
import llm.prepare as prepare
import llm.client as client
import rb_client as rb_client
import test_client

djl_version = os.environ.get('TEST_DJL_VERSION', '').strip()

def is_applicable_cuda_capability(arch: int) -> bool:
    import torch
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    return (10 * major + minor) >= arch


class Runner:

    def __init__(self, container, test_name=None, download=False):
        self.container = container
        self.test_name = test_name
        self.client_file_handler = None

        # Compute flavor and repo
        repo = "deepjavalibrary/djl-serving"
        if djl_version is None or len(
                djl_version) == 0 or djl_version == "nightly":
            flavor = f"{container}-nightly"
        else:
            if djl_version == "temp":
                repo = "185921645874.dkr.ecr.us-east-1.amazonaws.com/djl-ci-temp"
                flavor = f"{container}-{os.environ['GITHUB_SHA']}"
            else:
                if container == "cpu":
                    flavor = djl_version
                else:
                    flavor = f"{djl_version}-{container}"

        self.image = f"{repo}:{flavor}"

        # os.system(f'docker pull {self.image}')
        os.system('rm -rf models')

        if download:
            os.system(f"./download_models.sh {self.container}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        client.remove_file_handler_from_logger(self.client_file_handler)
        if self.test_name is not None:
            esc_test_name = self.test_name.replace("/", "-")
            os.system(f"mkdir -p all_logs/{esc_test_name}")
            os.system(
                f"cp client_logs/{esc_test_name}_client.log all_logs/{esc_test_name}/ || true"
            )
            os.system(f"cp -r logs all_logs/{esc_test_name}")
        subprocess.run(["./remove_container.sh"],
                       check=True,
                       capture_output=True)
        os.system("cat logs/serving.log")

    def launch(self, env_vars=None, container=None, cmd=None):
        if env_vars is not None:
            if isinstance(env_vars, list):
                env_vars = "\n".join(env_vars)
            with open("docker_env", "w") as f:
                f.write(env_vars)
        else:
            if os.path.isfile("docker_env"):
                os.remove("docker_env")

        if container is None:
            container = self.container

        if cmd is None:
            cmd = 'serve -m test=file:/opt/ml/model/test/'

        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs("client_logs", exist_ok=True)
        if self.test_name:
            esc_test_name = self.test_name.replace("/", "-")
            self.client_file_handler = client.add_file_handler_to_logger(
                f"client_logs/{esc_test_name}_client.log")
        return subprocess.run(
            f'./launch_container.sh {self.image} {model_dir} {container} {cmd}'
            .split(),
            check=True,
            capture_output=True)


@pytest.mark.cpu
class TestCpuFull:

    def test_python_model(self):
        with Runner('cpu-full', 'python_model', download=True) as r:
            r.launch(
                cmd=
                "serve -m test::Python=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")
            os.system("./test_client.sh tensor/ndlist 1,3,224,224")
            os.system("./test_client.sh tensor/npz 1,3,224,224")

    def test_python_dynamic_batch(self):
        with Runner('cpu-full', 'dynamic_batch', download=True) as r:
            env = ["SERVING_BATCH_SIZE=2", "SERVING_MAX_BATCH_DELAY=30000"]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m test::Python=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system(
                "EXPECT_TIMEOUT=1 ./test_client.sh image/jpg models/kitten.jpg"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")


@pytest.mark.cpu
@pytest.mark.parametrize('arch', ["cpu", "cpu-full"])
class TestCpuBoth:

    def test_pytorch(self, arch):
        with Runner(arch, 'pytorch', download=True) as r:
            r.launch(
                cmd=
                "serve -m test::PyTorch=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_pytorch_binary(self, arch):
        with Runner(arch, 'pytorch_binary', download=True) as r:
            r.launch(
                cmd=
                'serve -m "test::PyTorch=file:/opt/ml/model/resnet18_all_batch.zip?translatorFactory=ai.djl.translate.NoopServingTranslatorFactory&application=undefined'
            )
            os.system("./test_client.sh tensor/ndlist 1,3,224,224")
            os.system("./test_client.sh tensor/npz 1,3,224,224")

    def test_pytorch_dynamic_batch(self, arch):
        with Runner(arch, 'pytorch_dynamic_batch', download=True) as r:
            env = ["SERVING_BATCH_SIZE=2", "SERVING_MAX_BATCH_DELAY=30000"]
            r.launch(
                env_vars=env,
                cmd=
                'serve -m "test::PyTorch=file:/opt/ml/model/resnet18_all_batch.zip?translatorFactory=ai.djl.translate.NoopServingTranslatorFactory&application=undefined'
            )
            os.system(
                "EXPECT_TIMEOUT=1 ./test_client.sh image/jpg models/kitten.jpg"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_mxnet(self, arch):
        with Runner(arch, 'mxnet', download=True) as r:
            r.launch(
                cmd='serve -m test::MXNet=file:/opt/ml/model/ssd_resnet50.zip')
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_onnx(self, arch):
        with Runner(arch, 'onnx', download=True) as r:
            r.launch(
                cmd=
                'serve -m test::OnnxRuntime=file:/opt/ml/model/resnet18-v1-7.zip'
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_tensorflow_binary(self, arch):
        with Runner(arch, 'tensorflow_binary', download=True) as r:
            r.launch(
                cmd=
                'serve -m test::TensorFlow=file:/opt/ml/model/resnet50v1.zip?model_name=resnet50'
            )
            os.system("./test_client.sh tensor/ndlist 1,224,224,3")


@pytest.mark.gpu
@pytest.mark.gpu_4
class TestGpu:

    def test_python_model(self):
        with Runner('pytorch-gpu', 'python_model', download=True) as r:
            r.launch(
                cmd=
                "serve -m test::Python=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_pytorch(self):
        with Runner('pytorch-gpu', 'pytorch_model', download=True) as r:
            r.launch(
                cmd=
                "serve -m test::PyTorch=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")


@pytest.mark.aarch64
class TestAarch64:

    def test_pytorch(self):
        with Runner('aarch64', 'pytorch_model', download=True) as r:
            r.launch(
                cmd=
                "serve -m test::PyTorch=file:/opt/ml/model/resnet18_all_batch.zip"
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")

    def test_onnx(self):
        with Runner('aarch64', 'onnx', download=True) as r:
            r.launch(
                cmd=
                'serve -m test::OnnxRuntime=file:/opt/ml/model/resnet18-v1-7.zip'
            )
            os.system("./test_client.sh image/jpg models/kitten.jpg")


@pytest.mark.hf
@pytest.mark.gpu_4
class TestHfHandler:

    def test_gpt_neo(self):
        with Runner('lmi', 'test_gpt_neo_2.7b') as r:
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

    def test_llama3_lora(self):
        with Runner('lmi', 'llama3-tiny-random-lora') as r:
            prepare.build_hf_handler_model("llama3-tiny-random-lora")
            r.launch()
            client.run("huggingface llama3-tiny-random-lora".split())

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


@pytest.mark.trtllm
@pytest.mark.gpu_4
class TestTrtLlmHandler1:

    def test_llama2_13b_tp4(self):
        with Runner('tensorrt-llm', 'llama2-13b') as r:
            prepare.build_trtllm_handler_model("llama2-13b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm llama2-13b".split())

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

    def test_llama_31_8b(self):
        with Runner('tensorrt-llm', 'llama-3-1-8b') as r:
            prepare.build_trtllm_handler_model('llama-3-1-8b')
            r.launch()
            client.run("trtllm llama-3-1-8b".split())


@pytest.mark.trtllm
@pytest.mark.gpu_4
class TestTrtLlmHandler2:

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

    def test_llama2_7b_chat(self):
        with Runner('tensorrt-llm', 'llama2-7b-chat') as r:
            prepare.build_trtllm_handler_model("llama2-7b-chat")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm_chat llama2-7b-chat".split())

    def test_flan_t5_xl(self):
        with Runner('tensorrt-llm', "flan-t5-xl") as r:
            prepare.build_trtllm_handler_model("flan-t5-xl")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("trtllm flan-t5-xl".split())


@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestSchedulerSingleGPU:

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


@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestSchedulerMultiGPU:

    def test_gptj_6b(self):
        with Runner('lmi', 'gpt-j-6b') as r:
            prepare.build_rolling_batch_model("gpt-j-6b")
            r.launch()
            rb_client.run("scheduler_multi_gpu gpt-j-6b".split())


@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestLmiDist1:

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
            prepare.build_lmi_dist_model("gpt2")
            envs = [
                "OPTION_MAX_ROLLING_BATCH_SIZE=2",
                "OPTION_OUTPUT_FORMATTER=jsonlines",
                "TENSOR_PARALLEL_DEGREE=1", "OPTION_TASK=text-generation",
                "OPTION_ROLLING_BATCH=lmi-dist"
            ]
            r.launch("\n".join(envs))
            client.run("lmi_dist gpt2".split())

    def test_mpt_7b(self):
        with Runner('lmi', 'mpt-7b') as r:
            prepare.build_lmi_dist_model("mpt-7b")
            r.launch()
            client.run("lmi_dist mpt-7b".split())

    def test_mistral_7b_marlin(self):
        with Runner('lmi', 'mistral-7b-marlin') as r:
            prepare.build_lmi_dist_model("mistral-7b-marlin")
            r.launch()
            client.run("lmi_dist mistral-7b-marlin".split())

    def test_llama2_13b_flashinfer(self):
        with Runner('lmi', 'llama-2-13b-flashinfer') as r:
            prepare.build_lmi_dist_model("llama-2-13b-flashinfer")
            envs = [
                "VLLM_ATTENTION_BACKEND=FLASHINFER",
            ]
            r.launch(env_vars=envs)
            client.run("lmi_dist llama-2-13b-flashinfer".split())

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

    def test_llama3_8b_chunked_prefill(self):
        with Runner('lmi', 'llama3-8b-chunked-prefill') as r:
            prepare.build_lmi_dist_model("llama3-8b-chunked-prefill")
            r.launch()
            client.run(
                "lmi_dist llama3-8b-chunked-prefill --in_tokens 1200".split())

    def test_falcon_11b_chunked_prefill(self):
        with Runner('lmi', 'falcon-11b-chunked-prefill') as r:
            prepare.build_lmi_dist_model("falcon-11b-chunked-prefill")
            r.launch()
            client.run(
                "lmi_dist falcon-11b-chunked-prefill --in_tokens 1200".split())

    def test_flan_t5_xl(self):
        with Runner('lmi', 'flan-t5-xl') as r:
            prepare.build_lmi_dist_model("flan-t5-xl")
            r.launch()
            client.run("lmi_dist flan-t5-xl".split())


@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestLmiDist2:

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

    def test_llama31_8b_secure(self):
        with Runner('lmi', 'llama-3.1-8b') as r:
            prepare.build_lmi_dist_model("llama-3.1-8b")
            envs = [
                "SAGEMAKER_SECURE_MODE=True",
                "SAGEMAKER_SECURITY_CONTROLS=DISALLOW_CUSTOM_INFERENCE_SCRIPTS"
            ]
            r.launch(env_vars=envs)
            client.run("lmi_dist llama-3.1-8b".split())


@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestLmiDistMultiNode:

    def test_llama3_8b(self):
        with Runner('lmi', 'llama3-8b') as r:
            prepare.build_lmi_dist_model("llama3-8b")
            r.launch(cmd="multi_node")
            client.run("lmi_dist llama3-8b --in_tokens 1200".split())


@pytest.mark.vllm
@pytest.mark.gpu_4
class TestVllm1:

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

    @pytest.mark.skipif(not is_applicable_cuda_capability(89),
                        reason="Unsupported CUDA capability")
    def test_qwen2_7b_fp8(self):
        with Runner('lmi', 'qwen2-7b-fp8') as r:
            prepare.build_vllm_model("qwen2-7b-fp8")
            r.launch()
            client.run("vllm qwen2-7b-fp8".split())

    def test_llama3_8b_chunked_prefill(self):
        with Runner('lmi', 'llama3-8b-chunked-prefill') as r:
            prepare.build_vllm_model("llama3-8b-chunked-prefill")
            r.launch()
            client.run(
                "vllm llama3-8b-chunked-prefill --in_tokens 1200".split())

    def test_falcon_11b_chunked_prefill(self):
        with Runner('lmi', 'falcon-11b-chunked-prefill') as r:
            prepare.build_vllm_model("falcon-11b-chunked-prefill")
            r.launch()
            client.run(
                "vllm falcon-11b-chunked-prefill --in_tokens 1200".split())

    def test_llama_68m_speculative_medusa(self):
        with Runner('lmi', 'llama-68m-speculative-medusa') as r:
            prepare.build_vllm_model("llama-68m-speculative-medusa")
            r.launch()
            client.run("vllm llama-68m-speculative-medusa".split())

    def test_llama_68m_speculative_eagle(self):
        with Runner('lmi', 'llama-68m-speculative-eagle') as r:
            prepare.build_vllm_model("llama-68m-speculative-eagle")
            r.launch()
            client.run("vllm llama-68m-speculative-eagle".split())


@pytest.mark.vllm
@pytest.mark.lora
@pytest.mark.gpu_4
class TestVllmLora:

    def test_lora_llama2_7b(self):
        with Runner('lmi', 'llama-7b-unmerged-lora') as r:
            prepare.build_vllm_model("llama-7b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama-7b-unmerged-lora".split())

    def test_lora_llama2_7b_overflow(self):
        with Runner('lmi', 'llama-7b-unmerged-lora-overflow') as r:
            prepare.build_vllm_model("llama-7b-unmerged-lora-overflow")
            r.launch()
            client.run("vllm_adapters llama-7b-unmerged-lora-overflow".split())

    def test_lora_llama2_13b_awq(self):
        with Runner('lmi', 'llama2-13b-awq-unmerged-lora') as r:
            prepare.build_vllm_model("llama2-13b-awq-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama2-13b-awq-unmerged-lora".split())

    def test_lora_mistral_7b(self):
        with Runner('lmi', 'mistral-7b-unmerged-lora') as r:
            prepare.build_vllm_model("mistral-7b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters mistral-7b-unmerged-lora".split())

    def test_lora_mistral_7b_awq(self):
        with Runner('lmi', 'mistral-7b-awq-unmerged-lora') as r:
            prepare.build_vllm_model("mistral-7b-awq-unmerged-lora")
            r.launch()
            client.run("vllm_adapters mistral-7b-awq-unmerged-lora".split())

    def test_lora_mistral_7b_gptq(self):
        with Runner('lmi', 'mistral-7b-gptq-unmerged-lora') as r:
            prepare.build_lmi_dist_model("mistral-7b-gptq-unmerged-lora")
            r.launch()
            client.run("vllm_adapters mistral-7b-gptq-unmerged-lora".split())

    def test_lora_llama3_8b(self):
        with Runner('lmi', 'llama3-8b-unmerged-lora') as r:
            prepare.build_vllm_model("llama3-8b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters llama3-8b-unmerged-lora".split())

    def test_lora_gemma_7b(self):
        with Runner('lmi', 'gemma-7b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("gemma-7b-unmerged-lora")
            r.launch()
            client.run("vllm_adapters gemma-7b-unmerged-lora".split())

    def test_lora_phi2(self):
        with Runner('lmi', 'phi2-unmerged-lora') as r:
            prepare.build_lmi_dist_model("phi2-unmerged-lora")
            r.launch()
            client.run("vllm_adapters phi2-unmerged-lora".split())


@pytest.mark.lmi_dist
@pytest.mark.lora
@pytest.mark.gpu_4
class TestLmiDistLora:

    def test_lora_llama2_7b(self):
        with Runner('lmi', 'llama-7b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("llama-7b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters llama-7b-unmerged-lora".split())

    def test_lora_llama2_7b_overflow(self):
        with Runner('lmi', 'llama-7b-unmerged-lora-overflow') as r:
            prepare.build_lmi_dist_model("llama-7b-unmerged-lora-overflow")
            r.launch()
            client.run(
                "lmi_dist_adapters llama-7b-unmerged-lora-overflow".split())

    def test_lora_llama2_13b_awq(self):
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

    def test_lora_mistral_7b_awq(self):
        with Runner('lmi', 'mistral-7b-awq-unmerged-lora') as r:
            prepare.build_lmi_dist_model("mistral-7b-awq-unmerged-lora")
            r.launch()
            client.run(
                "lmi_dist_adapters mistral-7b-awq-unmerged-lora".split())

    def test_lora_mistral_7b_gptq(self):
        with Runner('lmi', 'mistral-7b-gptq-unmerged-lora') as r:
            prepare.build_lmi_dist_model("mistral-7b-gptq-unmerged-lora")
            r.launch()
            client.run(
                "lmi_dist_adapters mistral-7b-gptq-unmerged-lora".split())

    def test_lora_llama3_8b(self):
        with Runner('lmi', 'llama3-8b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("llama3-8b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters llama3-8b-unmerged-lora".split())

    def test_lora_gemma_7b(self):
        with Runner('lmi', 'gemma-7b-unmerged-lora') as r:
            prepare.build_lmi_dist_model("gemma-7b-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters gemma-7b-unmerged-lora".split())

    def test_lora_phi2(self):
        with Runner('lmi', 'phi2-unmerged-lora') as r:
            prepare.build_lmi_dist_model("phi2-unmerged-lora")
            r.launch()
            client.run("lmi_dist_adapters phi2-unmerged-lora".split())


@pytest.mark.inf
class TestNeuronx1:

    def test_python_mode(self):
        with Runner('pytorch-inf2', 'test_python_mode', download=True) as r:
            r.launch(
                cmd=
                'serve -m test::PyTorch:nc0=file:/opt/ml/model/resnet18_inf2_2_4.tar.gz',
                container='pytorch-inf2-1')
            test_client.run()

    def test_gpt2(self):
        with Runner('pytorch-inf2', 'gpt2') as r:
            prepare.build_transformers_neuronx_handler_model("gpt2")
            r.launch(container='pytorch-inf2-1')
            client.run("transformers_neuronx gpt2".split())

    def test_gpt2_quantize(self):
        with Runner('pytorch-inf2', 'gpt2-quantize') as r:
            prepare.build_transformers_neuronx_handler_model("gpt2-quantize")
            r.launch(container='pytorch-inf2-1')
            client.run("transformers_neuronx gpt2-quantize".split())

    @pytest.mark.parametrize(
        "model",
        ["tiny-llama-rb-aot", "tiny-llama-rb-aot-quant", "tiny-llama-rb-lcnc"])
    def test_partition(self, model):
        with Runner('pytorch-inf2', f'partition-{model}') as r:
            try:
                prepare.build_transformers_neuronx_handler_model(model)
                r.launch(
                    container="pytorch-inf2-1",
                    cmd=
                    "partition --model-dir /opt/ml/input/data/training --save-mp-checkpoint-path /opt/ml/input/data/training/aot --skip-copy"
                )
                r.launch(container="pytorch-inf2-1",
                         cmd="serve -m test=file:/opt/ml/model/test/aot")
                client.run(
                    "transformers_neuronx_rolling_batch tiny-llama-rb".split())
            finally:
                os.system('sudo rm -rf models')


@pytest.mark.inf
class TestNeuronx2:

    def test_stream_opt(self):
        with Runner('pytorch-inf2', 'opt-1.3b-streaming') as r:
            prepare.build_transformers_neuronx_handler_model(
                "opt-1.3b-streaming")
            r.launch(container='pytorch-inf2-6')
            client.run("transformers_neuronx opt-1.3b-streaming".split())

    def test_mixtral(self):
        with Runner('pytorch-inf2', 'mixtral-8x7b') as r:
            prepare.build_transformers_neuronx_handler_model("mixtral-8x7b")
            r.launch(container='pytorch-inf2-4')
            client.run("transformers_neuronx mixtral-8x7b".split())

    def test_stable_diffusion_1_5(self):
        with Runner('pytorch-inf2', 'stable-diffusion-1.5-neuron') as r:
            prepare.build_transformers_neuronx_handler_model(
                "stable-diffusion-1.5-neuron")
            r.launch(container='pytorch-inf2-2')
            client.run(
                "neuron-stable-diffusion stable-diffusion-1.5-neuron".split())

    def test_stable_diffusion_2_1(self):
        with Runner('pytorch-inf2', 'stable-diffusion-2.1-neuron') as r:
            prepare.build_transformers_neuronx_handler_model(
                "stable-diffusion-2.1-neuron")
            r.launch(container='pytorch-inf2-2')
            client.run(
                "neuron-stable-diffusion stable-diffusion-2.1-neuron".split())

    def test_stable_diffusion_xl(self):
        with Runner('pytorch-inf2', 'stable-diffusion-xl-neuron') as r:
            prepare.build_transformers_neuronx_handler_model(
                "stable-diffusion-xl-neuron")
            r.launch(container='pytorch-inf2-2')
            client.run(
                "neuron-stable-diffusion stable-diffusion-xl-neuron".split())


@pytest.mark.inf
class TestNeuronxRollingBatch:

    def test_llama_7b(self):
        with Runner('pytorch-inf2', 'llama-7b-rb') as r:
            prepare.build_transformers_neuronx_handler_model("llama-7b-rb")
            r.launch(container='pytorch-inf2-2')
            client.run(
                "transformers_neuronx_rolling_batch llama-7b-rb".split())

    def test_tiny_llama_vllm(self):
        with Runner('pytorch-inf2', 'tiny-llama-rb-vllm') as r:
            prepare.build_transformers_neuronx_handler_model(
                "tiny-llama-rb-vllm")
            r.launch(container='pytorch-inf2-1')
            client.run("transformers_neuronx_rolling_batch tiny-llama-rb-vllm".
                       split())

    def test_llama3_vllm(self):
        with Runner('pytorch-inf2', 'llama-3-8b-rb-vllm') as r:
            prepare.build_transformers_neuronx_handler_model(
                "llama-3-8b-rb-vllm")
            r.launch(container='pytorch-inf2-4')
            client.run("transformers_neuronx_rolling_batch llama-3-8b-rb-vllm".
                       split())

    def test_mistral(self):
        with Runner('pytorch-inf2', 'mistral-7b-rb') as r:
            prepare.build_transformers_neuronx_handler_model("mistral-7b-rb")
            r.launch(container='pytorch-inf2-2')
            client.run(
                "transformers_neuronx_rolling_batch mistral-7b-rb".split())

    def test_llama_speculative(self):
        with Runner('pytorch-inf2', 'llama-speculative-rb') as r:
            prepare.build_transformers_neuronx_handler_model(
                "llama-speculative-rb")
            r.launch(container='pytorch-inf2-6')
            client.run(
                "transformers_neuronx_rolling_batch llama-speculative-rb".
                split())

    def test_llama_speculative_compiled(self):
        with Runner('pytorch-inf2', 'llama-speculative-compiled-rb') as r:
            prepare.build_transformers_neuronx_handler_model(
                "llama-speculative-compiled-rb")
            r.launch(container='pytorch-inf2-6')
            client.run(
                "transformers_neuronx_rolling_batch llama-speculative-compiled-rb"
                .split())


@pytest.mark.correctness
@pytest.mark.trtllm
@pytest.mark.gpu_4
class TestCorrectnessTrtLlm:

    def test_codestral_22b(self):
        with Runner('tensorrt-llm', 'codestral-22b') as r:
            prepare.build_correctness_model("trtllm-codestral-22b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("correctness trtllm-codestral-22b".split())

    def test_llama3_8b(self):
        with Runner('tensorrt-llm', 'llama3-8b') as r:
            prepare.build_correctness_model("trtllm-llama3-8b")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("correctness trtllm-llama3-8b".split())

    def test_llama3_8b_fp8(self):
        with Runner('tensorrt-llm', 'llama3-3b') as r:
            prepare.build_correctness_model("trtllm-meta-llama3-8b-fp8")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("correctness trtllm-meta-llama3-8b-fp8".split())

    def test_mistral_7b(self):
        with Runner('tensorrt-llm', 'mistral-7b') as r:
            prepare.build_correctness_model("trtllm-mistral-7b-instruct-v0.3")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run("correctness trtllm-mistral-7b-instruct-v0.3".split())

    def test_mistral_7b_fp8(self):
        with Runner('tensorrt-llm', 'mistral-7b') as r:
            prepare.build_correctness_model(
                "trtllm-mistral-7b-instruct-v0.3-fp8")
            r.launch("CUDA_VISIBLE_DEVICES=0,1,2,3")
            client.run(
                "correctness trtllm-mistral-7b-instruct-v0.3-fp8".split())


@pytest.mark.correctness
@pytest.mark.lmi_dist
@pytest.mark.gpu_4
class TestCorrectnessLmiDist:

    def test_codestral_22b(self):
        with Runner('lmi', 'codestral-22b') as r:
            prepare.build_correctness_model("lmi-dist-codestral-22b")
            r.launch()
            client.run("correctness lmi-dist-codestral-22b".split())

    def test_llama3_1_8b(self):
        with Runner('lmi', 'llama3-1-8b') as r:
            prepare.build_correctness_model("lmi-dist-llama3-1-8b")
            r.launch()
            client.run("correctness lmi-dist-llama3-1-8b".split())


@pytest.mark.correctness
@pytest.mark.inf
class TestCorrectnessNeuronx:

    def test_codestral_22b(self):
        with Runner('pytorch-inf2', 'codestral-22b') as r:
            prepare.build_correctness_model("neuronx-codestral-22b")
            r.launch(container='pytorch-inf2-6')
            client.run("correctness neuronx-codestral-22b".split())

    def test_llama3_1_8b(self):
        with Runner('pytorch-inf2', 'llama3-1-8b') as r:
            prepare.build_correctness_model("neuronx-llama3-1-8b")
            r.launch(container='pytorch-inf2-6')
            client.run("correctness neuronx-llama3-1-8b".split())


class TestMultiModalLmiDist:

    def test_llava_next(self):
        with Runner('lmi', 'llava_v1.6-mistral') as r:
            prepare.build_lmi_dist_model('llava_v1.6-mistral')
            r.launch()
            client.run("multimodal llava_v1.6-mistral".split())

    def test_phi3_v(self):
        with Runner('lmi', 'phi-3-vision-128k-instruct') as r:
            prepare.build_lmi_dist_model('phi-3-vision-128k-instruct')
            r.launch()
            client.run("multimodal phi-3-vision-128k-instruct".split())

    def test_pixtral_12b(self):
        with Runner('lmi', 'pixtral-12b') as r:
            prepare.build_lmi_dist_model('pixtral-12b')
            r.launch()
            client.run("multimodal pixtral-12b".split())


class TestMultiModalVllm:

    def test_llava_next(self):
        with Runner('lmi', 'llava_v1.6-mistral') as r:
            prepare.build_vllm_model('llava_v1.6-mistral')
            r.launch()
            client.run("multimodal llava_v1.6-mistral".split())

    def test_phi3_v(self):
        with Runner('lmi', 'phi-3-vision-128k-instruct') as r:
            prepare.build_vllm_model('phi-3-vision-128k-instruct')
            r.launch()
            client.run("multimodal phi-3-vision-128k-instruct".split())

    def test_pixtral_12b(self):
        with Runner('lmi', 'pixtral-12b') as r:
            prepare.build_vllm_model('pixtral-12b')
            r.launch()
            client.run("multimodal pixtral-12b".split())

    # MLlama is only supported by vllm backend currently
    def test_mllama_11b(self):
        with Runner('lmi', 'llama32-11b-multimodal') as r:
            prepare.build_vllm_model('llama32-11b-multimodal')
            r.launch()
            client.run("multimodal llama32-11b-multimodal".split())


class TestLmiDistPipelineParallel:

    def test_llama32_3b_multi_worker_tp1_pp1(self):
        with Runner('lmi', 'llama32-3b-multi-worker-tp1-pp1') as r:
            prepare.build_lmi_dist_model("llama32-3b-multi-worker-tp1-pp1")
            r.launch()
            client.run("lmi_dist llama32-3b-multi-worker-tp1-pp1".split())

    def test_llama32_3b_multi_worker_tp2_pp1(self):
        with Runner('lmi', 'llama32-3b-multi-worker-tp2-pp1') as r:
            prepare.build_lmi_dist_model("llama32-3b-multi-worker-tp2-pp1")
            r.launch()
            client.run("lmi_dist llama32-3b-multi-worker-tp2-pp1".split())

    def test_llama32_3b_multi_worker_tp1_pp2(self):
        with Runner('lmi', 'llama32-3b-multi-worker-tp1-pp2') as r:
            prepare.build_lmi_dist_model("llama32-3b-multi-worker-tp1-pp2")
            r.launch()
            client.run("lmi_dist llama32-3b-multi-worker-tp1-pp2".split())

    def test_llama31_8b_pp_only(self):
        with Runner('lmi', 'llama31-8b-pp-only') as r:
            prepare.build_lmi_dist_model("llama31-8b-pp-only")
            r.launch()
            client.run("lmi_dist llama31-8b-pp-only".split())

    def test_llama31_8b_tp2_pp2(self):
        with Runner('lmi', 'llama31-8b-tp2-pp2') as r:
            prepare.build_lmi_dist_model('llama31-8b-tp2-pp2')
            r.launch()
            client.run("lmi_dist llama31-8b-tp2-pp2".split())

    def test_llama31_8b_tp2_pp2_specdec(self):
        with Runner('lmi', 'llama31-8b-tp2-pp2-spec-dec') as r:
            prepare.build_lmi_dist_model('llama31-8b-tp2-pp2-spec-dec')
            r.launch()
            client.run("lmi_dist llama31-8b-tp2-pp2-spec-dec".split())


@pytest.mark.gpu
class TestTextEmbedding:

    def test_bge_base_rust(self):
        with Runner('lmi', 'bge-base-rust') as r:
            prepare.build_text_embedding_model("bge-base-rust")
            r.launch()
            client.run("text_embedding bge-base-rust".split())

    def test_e5_base_v2_rust(self):
        with Runner('lmi', 'e5-base-v2-rust') as r:
            prepare.build_text_embedding_model("e5-base-v2-rust")
            r.launch()
            client.run("text_embedding e5-base-v2-rust".split())

    def test_sentence_camembert_large_rust(self):
        with Runner('lmi', 'sentence-camembert-large-rust') as r:
            prepare.build_text_embedding_model("sentence-camembert-large-rust")
            r.launch()
            client.run("text_embedding sentence-camembert-large-rust".split())

    def test_roberta_base_rust(self):
        with Runner('lmi', 'roberta-base-rust') as r:
            prepare.build_text_embedding_model("roberta-base-rust")
            r.launch()
            client.run("text_embedding roberta-base-rust".split())

    def test_msmarco_distilbert_base_v4_rust(self):
        with Runner('lmi', 'msmarco-distilbert-base-v4-rust') as r:
            prepare.build_text_embedding_model(
                "msmarco-distilbert-base-v4-rust")
            r.launch()
            client.run(
                "text_embedding msmarco-distilbert-base-v4-rust".split())

    def test_bge_reranker_rust(self):
        with Runner('lmi', 'bge-reranker-rust') as r:
            prepare.build_text_embedding_model("bge-reranker-rust")
            r.launch()
            client.run("text_embedding bge-reranker-rust".split())

    def test_e5_mistral_7b_rust(self):
        with Runner('lmi', 'e5-mistral-7b-rust') as r:
            prepare.build_text_embedding_model("e5-mistral-7b-rust")
            r.launch()
            client.run("text_embedding e5-mistral-7b-rust".split())

    def test_gte_qwen2_7b_rust(self):
        with Runner('lmi', 'gte-qwen2-7b-rust') as r:
            prepare.build_text_embedding_model("gte-qwen2-7b-rust")
            r.launch()
            client.run("text_embedding gte-qwen2-7b-rust".split())

    def test_gte_large_rust(self):
        with Runner('lmi', 'gte-large-rust') as r:
            prepare.build_text_embedding_model("gte-large-rust")
            r.launch()
            client.run("text_embedding gte-large-rust".split())

    def test_bge_multilingual_gemma2_rust(self):
        with Runner('lmi', 'bge-multilingual-gemma2-rust') as r:
            prepare.build_text_embedding_model("bge-multilingual-gemma2-rust")
            r.launch()
            client.run("text_embedding bge-multilingual-gemma2-rust".split())

    def test_bge_base_onnx(self):
        with Runner('lmi', 'bge-base-onnx') as r:
            prepare.build_text_embedding_model("bge-base-onnx")
            r.launch()
            client.run("text_embedding bge-base-onnx".split())


@pytest.mark.gpu
@pytest.mark.handler_performance
class TestGPUHandlerPerformance:

    def test_lmi_dist(self):
        with Runner('lmi', 'handler-performance-lmi-dist') as r:
            prepare.build_handler_performance_model("tiny-llama-lmi")
            r.launch("CUDA_VISIBLE_DEVICES=0")
            client.run("handler_performance lmi".split())

    def test_vllm(self):
        with Runner('lmi', 'handler-performance-vllm') as r:
            prepare.build_handler_performance_model("tiny-llama-vllm")
            r.launch("CUDA_VISIBLE_DEVICES=0")
            client.run("handler_performance vllm".split())

    def test_trtllm(self):
        with Runner('tensorrt-llm', 'handler-performance-trtllm') as r:
            prepare.build_handler_performance_model("tiny-llama-trtllm")
            r.launch("CUDA_VISIBLE_DEVICES=0")
            client.run("handler_performance trtllm".split())
