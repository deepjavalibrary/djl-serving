#!/usr/bin/env python3

import os
import subprocess
import llm.prepare as prepare
import llm.client as client
import rb_client

djl_version = ''


class Runner:

    def __init__(self, container):
        self.container = container
        flavor = subprocess.run([
            '../../serving/docker/scripts/docker_name_builder.sh', container,
            djl_version
        ],
                                capture_output=True,
                                text=True).stdout.strip()
        self.image = f"deepjavalibrary/djl-serving:{flavor}"

    def __enter__(self):
        # os.system(f'docker pull {self.image}')
        os.system('rm -rf models')
        return self

    def __exit__(self, *args):
        container = subprocess.run(['docker', 'ps', '-aq'],
                                   capture_output=True,
                                   text=True).stdout.strip()
        if container != '':
            subprocess.run(['docker', 'rm', '-f', container],
                           shell=True,
                           check=True)
        os.system("cat logs/serving.log")

    def launch(self, env_vars=None):
        if env_vars is not None:
            with open("docker_env", "w") as f:
                f.write(env_vars)

        model_dir = os.path.join(os.getcwd(), 'models')
        subprocess.run(
            f'./launch_container.sh {self.image} {model_dir} {self.container} serve -m test=file:/opt/ml/model/test/'
            .split(),
            check=True)


def test_gpt_neo():
    with Runner('deepspeed') as r:
        prepare.build_hf_handler_model("gpt-neo-2.7b")
        r.launch()
        client.run("huggingface gpt-neo-2.7b".split())


def test_llama_7b():
    with Runner('deepspeed') as r:
        prepare.build_hf_handler_model("open-llama-7b")
        r.launch()
        client.run("huggingface open-llama-7b".split())


def test_unmerged_lora_llama7b():
    with Runner('deepspeed') as r:
        prepare.build_hf_handler_model("llama-7b-unmerged-lora")
        r.launch()
        client.run("huggingface llama-7b-unmerged-lora".split())


def test_falcon_7b_triton_tp1():
    with Runner('tensorrt-llm') as r:
        prepare.build_trtllm_handler_model("falcon-7b")
        r.launch("CUDA_VISIBLE_DEVICES=0")
        client.run("trtllm falcon-7b".split())


def test_bloom_560m():
    with Runner('deepspeed') as r:
        prepare.build_rolling_batch_model("bloom-560m")
        r.launch()
        rb_client.run("scheduler_single_gpu bloom-560m".split())
