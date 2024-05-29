import sys
import requests
import argparse
import subprocess as sp
import logging
import re
import os
import math
import json
import shutil
from random import randrange
import numpy as np
from datetime import datetime
from io import BytesIO

logging.basicConfig(level=logging.INFO)


def get_model_name():
    endpoint = f"http://127.0.0.1:8080/models"
    res = requests.get(endpoint).json()
    return res["models"][0]["modelName"]


hf_model_spec = {
    "gpt-neo-2.7b": {
        "max_memory_per_gpu": [8.0, 8.0, 9.0, 17.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "gpt-j-6b": {
        "max_memory_per_gpu": [8.0, 9.0, 9.0, 21.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "llama-2-7b": {
        "max_memory_per_gpu": [10.0, 7.0, 7.0, 17.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256]
    },
    "bloom-7b1": {
        "max_memory_per_gpu": [7.0, 7.0, 8.0, 9.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128]
    },
    "bigscience/bloom-3b": {
        "max_memory_per_gpu": [5.0, 6.0],
        "batch_size": [1, 4],
        "seq_length": [16, 32],
        "worker": 1,
        "stream_output": True,
    },
    "t5-large": {
        "max_memory_per_gpu": [5.0],
        "batch_size": [1],
        "seq_length": [32],
        "worker": 1,
        "stream_output": True,
    },
    "gpt4all-lora": {
        "max_memory_per_gpu": [10.0, 12.0],
        "batch_size": [1, 4],
        "seq_length": [16, 32],
        "worker": 1,
    }
}

neuron_sd_model_spec = {
    "stable-diffusion-1.5-neuron": {
        "num_inference_steps": [50, 100]
    },
    "stable-diffusion-2.1-neuron": {
        "num_inference_steps": [50, 100]
    },
    "stable-diffusion-xl-neuron": {
        "num_inference_steps": [50, 100]
    }
}

transformers_neuronx_model_spec = {
    "gpt2": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [4]
    },
    "gpt2-quantize": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [4]
    },
    "opt-1.3b": {
        "worker": 3,
        "seq_length": [128, 256],
        "batch_size": [4]
    },
    "pythia-2.8b": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [4],
        "use_sample": True
    },
    "open-llama-7b": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [4],
        "use_sample": True
    },
    "llama-7b-split": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [1],
    },
    "bloom-7b1": {
        "worker": 1,
        "seq_length": [128],
        "batch_size": [4]
    },
    "gpt-j-6b": {
        "worker": 1,
        "seq_length": [128, 256, 512],
        "batch_size": [4]
    },
    "opt-1.3b-streaming": {
        "worker": 3,
        "seq_length": [128, 256],
        "batch_size": [2],
        "stream_output": True,
    },
    "mistral-7b": {
        "worker": 1,
        "seq_length": [128, 256],
        "batch_size": [4],
    },
    "mistral-7b-rb": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "mixtral-8x7b-rb": {
        "batch_size": [4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "llama-7b-rb": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama-3-8b-rb-vllm": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "tiny-llama-rb-vllm": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama-speculative-rb": {
        "batch_size": [1],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama-speculative-compiled-rb": {
        "batch_size": [1],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    }
}

transformers_neuronx_aot_model_spec = {
    "gpt2": {
        "worker": 1,
        "seq_length": [512],
        "batch_size": [4]
    },
    "gpt2-quantize": {
        "worker": 1,
        "seq_length": [512],
        "batch_size": [4]
    },
}

lmi_dist_model_spec = {
    "gpt-neox-20b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b"
    },
    "falcon-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-7b"
    },
    "falcon-11b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-11b"
    },
    "flan-t5-xxl": {
        "max_memory_per_gpu": [10.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "google/flan-t5-xxl"
    },
    "gpt2": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "gpt2"
    },
    "mpt-7b": {
        "max_memory_per_gpu": [20.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "mosaicml/mpt-7b"
    },
    "octocoder": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/octocoder"
    },
    "speculative-llama-13b": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "starcoder2-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/starcoder2-7b"
    },
    "gemma-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256]
    },
    "llama2-13b-gptq": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
        "parameters": {
            "decoder_input_details": True
        }
    },
    "mistral-7b": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "llama2-7b-32k": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [1024],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
        "parameters": {
            "decoder_input_details": True
        }
    },
    "mistral-7b-128k-awq": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "llama-7b-unmerged-lora": {
        "max_memory_per_gpu": [15.0, 15.0],
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["english-alpaca", "portugese-alpaca", "english-alpaca"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama2-13b-awq-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "mistralai/Mistral-7B-v0.1"
    },
    "mistral-7b-awq-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "mistralai/Mistral-7B-v0.1"
    },
    "llama-7b-unmerged-lora-overflow": {
        "max_memory_per_gpu": [15.0, 15.0],
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": [f"english-alpaca-{i}" for i in range(20)],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
}

lmi_dist_chat_model_spec = {
    "llama2-7b-chat": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    }
}

vllm_model_spec = {
    "gpt-neox-20b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b"
    },
    "llama2-13b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
        "parameters": {
            "decoder_input_details": True
        }
    },
    "phi-2": {
        "max_memory_per_gpu": [23.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "microsoft/phi-2"
    },
    "llama2-70b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mixtral-8x7b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "llama-7b-unmerged-lora": {
        "max_memory_per_gpu": [15.0, 15.0],
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["english-alpaca", "portugese-alpaca", "english-alpaca"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama2-13b-awq-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "mistralai/Mistral-7B-v0.1"
    },
    "mistral-7b-awq-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "mistralai/Mistral-7B-v0.1"
    },
    "llama-7b-unmerged-lora-overflow": {
        "max_memory_per_gpu": [15.0, 15.0],
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": [f"english-alpaca-{i}" for i in range(20)],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "starcoder2-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/starcoder2-7b"
    },
    "gemma-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256]
    },
}

vllm_chat_model_spec = {
    "llama2-7b-chat": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    }
}

lmi_dist_aiccl_model_spec = {
    "llama-2-70b-aiccl": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "codellama-34b-aiccl": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "codellama/CodeLlama-34b-hf"
    },
    "falcon-40b-aiccl": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-40b"
    },
    "mixtral-8x7b-aiccl": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
}

trtllm_model_spec = {
    "llama2-13b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "falcon-7b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-7b"
    },
    "llama2-7b-smoothquant": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "internlm-7b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "internlm/internlm-7b"
    },
    "baichuan2-13b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "chatglm3-6b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "mistral-7b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "gpt-j-6b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-j-6b"
    },
    "qwen-7b": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "Qwen/Qwen-7B"
    },
    "gpt2": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "gpt2"
    },
    "santacoder": {
        "max_memory_per_gpu": [22.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/santacoder"
    },
    "llama2-70b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mixtral-8x7b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "flan-t5-xl": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "google/flan-t5-xl",
        "details": True
    },
    "flan-t5-xxl": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "google/flan-t5-xxl"
    }
}

no_code_rolling_batch_spec = {
    "llama-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-fp16",
    },
    "llama-13b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
    },
    "gemma-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "mistral-7b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
    },
    "gpt-neox": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b",
    },
    "phi-2": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "microsoft/phi-2",
    },
    "baichuan-13b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "baichuan-inc/Baichuan2-13B-Base",
    },
    "qwen-1.5-14b": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "Qwen/Qwen1.5-14B",
    },
    "starcoder": {
        "max_memory_per_gpu": [25.0],
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama-70b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-70B-fp16",
    },
    "codellama": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "codellama/CodeLlama-34b-hf",
    },
    "mixtral-8x7b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    },
    "falcon-40b": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-40b",
    },
    "dbrx": {
        "max_memory_per_gpu": [40.0],
        "batch_size": [1, 4],
        "seq_length": [256],
    }
}


def check_worker_number(desired):
    model_name = get_model_name()
    endpoint = f"http://127.0.0.1:8080/models/{model_name}"
    res = requests.get(endpoint).json()
    if desired == len(res[0]["models"][0]["workerGroups"]):
        return
    elif desired == len(res[0]["models"][0]["workerGroups"][0]["workers"]):
        return
    else:
        raise AssertionError(
            f"Worker number does not meet requirements! {res}")


def send_json(data):
    headers = {'content-type': 'application/json'}
    endpoint = f"http://127.0.0.1:8080/invocations"
    resp = requests.post(endpoint, headers=headers, json=data)

    if resp.status_code >= 300:
        logging.exception(f"HTTP error: {resp}")
        raise ValueError("Failed to send reqeust to model server")
    return resp


def find_awscurl():
    command = "./awscurl -h"
    try:
        sp.check_output(command, shell=True)
    except sp.CalledProcessError:
        logging.info("Downloading awscurl...")
        command = "wget https://publish.djl.ai/awscurl/awscurl && chmod +x awscurl"
        sp.call(command, shell=True)


def awscurl_run(data, tokenizer, concurrency, num_run=5, dataset=False):
    find_awscurl()
    headers = "Content-type: application/json"
    endpoint = f"http://127.0.0.1:8080/invocations"
    if dataset:
        dataset_dir = os.path.join(os.path.curdir, "dataset")
        os.mkdir(dataset_dir)
        for i, d in enumerate(data):
            with open(os.path.join(dataset_dir, f"prompt{i}.txt"), "w") as f:
                f.write(json.dumps(d))
        command_data = f"--dataset {dataset_dir}"
    else:
        json_data = json.dumps(data)
        command_data = f"-d '{json_data}'"
    command = (f"./awscurl -c {concurrency} "
               f"-N {num_run} -X POST {endpoint} --connect-timeout 120 "
               f"-H {headers} {command_data} -P -t")
    if tokenizer:
        command = f"TOKENIZER={tokenizer} {command}"
    logging.info(f"Running command {command}")
    sp.call(command, shell=True)
    if dataset:
        shutil.rmtree(dataset_dir)


def send_image_json(img_url, data):
    multipart_form_data = {
        'data': BytesIO(requests.get(img_url, stream=True).content),
        'json': (None, json.dumps(data), 'application/json')
    }
    endpoint = f"http://127.0.0.1:8080/invocations"
    resp = requests.post(endpoint, files=multipart_form_data)

    if resp.status_code >= 300:
        logging.exception(f"HTTP error: {resp}")
        raise ValueError("Failed to send reqeust to model server")
    return resp


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]

    def convert_str_to_mem_used_gb(mem_free_info):
        return float(mem_free_info.split()[0]) / 1024.0

    return [
        convert_str_to_mem_used_gb(x) for i, x in enumerate(memory_free_info)
    ]


def validate_memory_usage(expected_memory_limit):
    used_memory_per_gpu = get_gpu_memory()
    logging.info(f"Used memory per GPU: {used_memory_per_gpu}")
    if any(x > expected_memory_limit for x in used_memory_per_gpu):
        raise AssertionError(f"Memory usage is too high!"
                             f"Used Memory:{used_memory_per_gpu}"
                             f"Expected Upper Limit:{expected_memory_limit}")


def fake_tokenizer(prompt, in_tokens):
    tokenized = re.findall(r"[\w']+|[.,!?;]", prompt)
    index_pointer = 0
    token_count = 0
    for token in tokenized:
        target = token[-1]
        index_pointer = prompt.find(target, index_pointer) + 1
        token_count += 1
        if token_count == in_tokens:
            break
    return prompt[:index_pointer]


def prompt_generation(in_tokens):
    with open(os.path.join(os.getcwd(), 'prompts.txt')) as f:
        result = '\n'.join(f.readlines())
    rot = result.find('. ', randrange(len(result))) + 2
    result = result[rot:] + result[:rot]

    return fake_tokenizer(result, in_tokens)


def batch_generation(batch_size):
    if args.in_tokens:
        input_sentences = [
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens)
        ]
    else:
        input_sentences = [
            "DeepSpeed is a machine learning framework",
            "He is working on",
            "He has a",
            "He got all",
            "Everyone is happy and I can",
            "The new movie that got Oscar this year",
            "In the far far distance from our galaxy,",
            "Peace is the only way",
        ]
    if batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    return input_sentences[:batch_size]


def batch_generation_chat(batch_size):
    messages = [
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
    ]

    if batch_size > len(messages):
        # dynamically extend to support larger bs by repetition
        messages *= math.ceil(batch_size / len(messages))
    return messages[:batch_size]


def t5_batch_generation(batch_size):
    input_sentences = [
        "translate English to German: The house is wonderful.",
        "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught \
             of severe weather in mississippi…",
    ]
    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    return input_sentences[:batch_size]


def get_total_memory(memory_snapshot):
    memory_footprint = 0
    for memory in memory_snapshot:
        memory_footprint += float(memory)
    return memory_footprint


def build_metric_label():
    sanitized_model_name = args.model.split("/")[-1]
    dtype = f"{args.dtype}"
    tp = f"tp-{args.tensor_parallel}" if args.tensor_parallel else ""
    batch_size = f"batch-{args.batch_size}" if args.batch_size else ""
    in_tokens = f"{args.in_tokens}-in-tokens" if args.in_tokens else ""
    out_tokens = f"{args.out_tokens}-out-tokens" if args.out_tokens else ""
    output = [
        sanitized_model_name, dtype, tp, batch_size, in_tokens, out_tokens
    ]
    while "" in output:
        output.remove("")
    return "_".join(output)


def log_metrics(response_times):
    required_args = ["batch_size", "out_tokens"]
    for arg in required_args:
        if arg not in args:
            raise ValueError(
                f"Logging metrics requires the following arguments: {required_args}"
            )

    p50 = np.percentile(response_times, 50)
    p90 = np.percentile(response_times, 90)
    throughput = 1000 / (sum(response_times) / len(response_times))
    tps = throughput * args.out_tokens * args.batch_size
    max_memory = get_total_memory(get_gpu_memory())

    outputs = []
    metric_stem = build_metric_label()
    outputs.append({
        "MetricName": f"{metric_stem}_p50",
        "Unit": "Milliseconds",
        "Value": p50
    })
    outputs.append({
        "MetricName": f"{metric_stem}_p90",
        "Unit": "Milliseconds",
        "Value": p90
    })
    outputs.append({
        "MetricName": f"{metric_stem}_throughput",
        "Unit": "Count/Second",
        "Value": throughput
    })
    outputs.append({
        "MetricName": f"{metric_stem}_tokens-per-second",
        "Unit": "Count/Second",
        "Value": tps
    })
    outputs.append({
        "MetricName": f"{metric_stem}_gpu-memory",
        "Unit": "Megabytes",
        "Value": max_memory
    })
    if args.cpu_memory > 0:
        outputs.append({
            "MetricName": f"{metric_stem}_cpu-memory",
            "Unit": "Kilobytes",
            "Value": args.cpu_memory
        })
    with open("llm/metrics.log", "w") as f:
        f.write(str(outputs))
        f.close()


def response_checker(res, message):
    if 'content-type' in res.headers.keys():
        if 'application/json' == res.headers['content-type']:
            output_json = json.loads(message)
            if isinstance(output_json, dict):
                if "details" in output_json.keys():
                    if "error" == output_json["details"]["finish_reason"]:
                        raise RuntimeError(f"Inference failed!")
                elif output_json.get("code", 200) != 200:
                    raise RuntimeError("Inference failed!")
        elif 'application/jsonlines' == res.headers['content-type']:
            json_lines = []
            for item in message.splitlines():
                try:
                    json_lines.append(json.loads(item))
                except:
                    raise RuntimeError(f"Json loading failure {item}")

            output_json = json_lines[-1]
            if "details" in output_json.keys():
                if "error" == output_json["details"]["finish_reason"]:
                    raise RuntimeError(f"Inference failed!")
            elif output_json.get("code", 200) != 200:
                raise RuntimeError("Inference failed!")
        else:
            logging.info(
                f"Skipping content check given non-supported content type {res.headers['content-type']}"
            )


def test_handler_rolling_batch(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    # dryrun phase
    req = {"inputs": batch_generation(1)[0]}
    seq_length = 100
    params = {"do_sample": True, "max_new_tokens": seq_length, "details": True}
    req["parameters"] = params
    if "parameters" in spec:
        req["parameters"].update(spec["parameters"])
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]
    logging.info(f"req {req}")
    res = send_json(req)
    message = res.content.decode("utf-8")
    logging.info(f"res: {message}")
    response_checker(res, message)

    # awscurl little benchmark phase
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            logging.info(
                f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
            )
            req["parameters"]["max_new_tokens"] = seq_length
            awscurl_run(req, spec.get("tokenizer", None), batch_size)


def test_handler_adapters(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    # dryrun phase
    reqs = []
    inputs = batch_generation(len(spec.get("adapters")))
    for i, adapter in enumerate(spec.get("adapters")):
        req = {"inputs": inputs[i]}
        seq_length = 100
        params = {
            "do_sample": True,
            "max_new_tokens": seq_length,
            "details": True
        }
        req["parameters"] = params
        req["adapters"] = adapter
        reqs.append(req)
    logging.info(f"reqs {reqs}")
    for req in reqs:
        res = send_json(req)
        message = res.content.decode("utf-8")
        logging.info(f"res: {message}")
        response_checker(res, message)
    # awscurl little benchmark phase
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            logging.info(
                f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
            )
            for req in reqs:
                req["parameters"]["max_new_tokens"] = seq_length
            awscurl_run(reqs,
                        spec.get("tokenizer", None),
                        batch_size,
                        dataset=True)
    # Test removing and querying invalid/removed adapter
    del_adapter = spec.get("adapters")[0]
    res = requests.delete(
        f"http://127.0.0.1:8080/models/test/adapters/{del_adapter}")
    logging.info(f"del adapter {res}")
    res = send_json(reqs[0]).content.decode("utf-8")
    logging.info(f"call deleted adapter {res}")
    if "error" not in res:
        raise RuntimeError(f"Should not work with new adapters")

    if len(reqs) > 1:
        res = send_json(reqs[1]).content.decode("utf-8")
        logging.info(f"call valid adapter after deletion {res}")
        if "error" in res:
            raise RuntimeError(f"Deleting adapter breaking inference")


def test_handler_rolling_batch_chat(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    # dryrun phase
    req = {"messages": batch_generation_chat(1)[0]}
    seq_length = 100
    req["max_tokens"] = seq_length
    req["logprobs"] = True
    req["top_logprobs"] = 1
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]
    logging.info(f"req {req}")
    res = send_json(req)
    logging.info(f"res: {res.content}")
    # awscurl little benchmark phase
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            logging.info(
                f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
            )
            req["max_tokens"] = seq_length
            awscurl_run(req, spec.get("tokenizer", None), batch_size)


def test_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            if "t5" in model:
                req = {"inputs": t5_batch_generation(batch_size)}
            else:
                req = {"inputs": batch_generation(batch_size)}
            if spec.get("adapters", []):
                req["adapters"] = spec.get("adapters")
            params = {"max_new_tokens": seq_length}
            if spec.get("details", False):
                params["details"] = True
            req["parameters"] = params
            logging.info(f"req {req}")
            res = send_json(req)
            if spec.get("stream_output", False):
                logging.info(f"res: {res.content}")
                result = res.content.decode().split("\n")[:-1]
                assert len(
                    result
                ) <= seq_length, "generated more tokens than max_new_tokens"
            else:
                res = res.json()
                logging.info(f"res {res}")
                if isinstance(res, list):
                    result = [item['generated_text'] for item in res]
                    assert len(result) == batch_size
                elif isinstance(res, dict):
                    assert 1 == batch_size
            if "max_memory_per_gpu" in spec:
                validate_memory_usage(spec["max_memory_per_gpu"][i])
            if "tokenizer" in spec:
                awscurl_run(req, spec.get("tokenizer"), batch_size)


def test_ds_raw_model(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            req = {
                "batch_size": batch_size,
                "text_length": seq_length,
                "use_pipeline": spec["use_pipeline"]
            }
            logging.info(f"req: {req}")
            res = send_json(req)
            res = res.json()
            logging.info(f"res: {res}")
            assert len(res["outputs"]) == batch_size
            if "max_memory_per_gpu" in spec:
                validate_memory_usage(spec["max_memory_per_gpu"][i])


def test_performance():
    response_times = []
    for i in range(args.count):
        req = {"inputs": batch_generation(args.batch_size)}
        params = {"max_new_tokens": args.out_tokens}
        req["parameters"] = params
        logging.info(f"req: {req}")
        start = datetime.now()
        res = send_json(req)
        delta = (datetime.now() - start).total_seconds() * 1000
        response_times.append(delta)
        res = res.json()
        logging.info(f"res: {res}")
    log_metrics(response_times)


def test_neuron_sd_handler(model, model_spec):
    from PIL import Image
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(neuron_sd_model_spec.keys())}"
        )
    spec = neuron_sd_model_spec[model]
    for step in spec["num_inference_steps"]:
        req = {"prompt": "A bird and cat flying through space"}
        params = {"num_inference_steps": step}
        req["parameters"] = params
        logging.info(f"req: {req}")
        res = send_json(req)
        try:
            Image.open(BytesIO(res.content)).convert("RGB")
        except Exception as e:
            raise IOError("failed to deserialize image from response", e)


def test_transformers_neuronx_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for batch_size in spec["batch_size"]:
        for seq_length in spec["seq_length"]:
            req = {"inputs": batch_generation(batch_size)}
            params = {"max_length": seq_length}
            if "use_sample" in spec:
                params["use_sample"] = True
            req["parameters"] = params
            logging.info(f"req {req}")
            res = send_json(req)
            if spec.get("stream_output", False):
                logging.info(f"res: {res.content}")
            else:
                res = res.json()
                logging.info(f"res {res}")
                result = res
                assert len(result) == batch_size


def run(raw_args):
    parser = argparse.ArgumentParser(description="Build the LLM configs")
    parser.add_argument("handler", help="the handler used in the model")
    parser.add_argument("model", help="The name of model")
    parser.add_argument("--engine",
                        required=False,
                        type=str,
                        choices=["deepspeed", "huggingface"],
                        help="The engine used for inference")
    parser.add_argument("--dtype",
                        required=False,
                        type=str,
                        help="The model data type")
    parser.add_argument("--tensor_parallel",
                        required=False,
                        type=int,
                        help="The model tensor parallel degree")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        help="The batch size of inference requests")
    parser.add_argument("--in_tokens",
                        required=False,
                        type=int,
                        help="The sequence length for input tokens")
    parser.add_argument("--out_tokens",
                        required=False,
                        type=int,
                        help="The sequence length for output tokens")
    parser.add_argument("--count",
                        required=False,
                        type=int,
                        help="Number of requests sent")
    parser.add_argument("--cpu_memory",
                        required=False,
                        default=0,
                        type=int,
                        help="CPU Memory footprint")
    global args
    args = parser.parse_args(args=raw_args)

    if args.handler == "huggingface":
        test_handler(args.model, hf_model_spec)
    elif args.handler == "neuron-stable-diffusion":
        test_neuron_sd_handler(args.model, neuron_sd_model_spec)
    elif args.handler == "transformers_neuronx":
        test_transformers_neuronx_handler(args.model,
                                          transformers_neuronx_model_spec)
    elif args.handler == "transformers_neuronx_rolling_batch":
        test_handler_rolling_batch(args.model, transformers_neuronx_model_spec)
    elif args.handler == "transformers_neuronx-aot":
        test_transformers_neuronx_handler(args.model,
                                          transformers_neuronx_aot_model_spec)
    elif args.handler == "lmi_dist":
        test_handler_rolling_batch(args.model, lmi_dist_model_spec)
    elif args.handler == "lmi_dist_adapters":
        test_handler_adapters(args.model, lmi_dist_model_spec)
    elif args.handler == "vllm":
        test_handler_rolling_batch(args.model, vllm_model_spec)
    elif args.handler == "vllm_adapters":
        test_handler_adapters(args.model, vllm_model_spec)
    elif args.handler == "lmi_dist_chat":
        test_handler_rolling_batch_chat(args.model, lmi_dist_chat_model_spec)
    elif args.handler == "vllm_chat":
        test_handler_rolling_batch_chat(args.model, vllm_chat_model_spec)
    elif args.handler == "performance":
        test_performance()
    elif args.handler == "lmi_dist_aiccl":
        test_handler_rolling_batch(args.model, lmi_dist_aiccl_model_spec)
    elif args.handler == "trtllm":
        test_handler_rolling_batch(args.model, trtllm_model_spec)
    elif args.handler == "trtllm-python":
        test_handler(args.model, trtllm_model_spec)
    elif args.handler == "no_code":
        test_handler_rolling_batch(args.model, no_code_rolling_batch_spec)

    else:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler")


if __name__ == "__main__":
    run(sys.argv[1:])
