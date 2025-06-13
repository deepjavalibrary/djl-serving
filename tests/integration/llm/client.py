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
import urllib

from concurrent.futures import ThreadPoolExecutor, as_completed
from json.decoder import JSONDecodeError

FAILED_DEPENDENCY_CODE = 424
TIMEOUT = 3.0
N_WORKERS = 8

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
# write output to console
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def get_model_name():
    endpoint = f"http://127.0.0.1:8080/models"
    res = requests.get(endpoint).json()
    return res["models"][0]["modelName"]


hf_model_spec = {
    "gpt-neo-2.7b": {
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "gpt-j-6b": {
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "llama-2-7b": {
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256]
    },
    "bloom-7b1": {
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128]
    },
    "bigscience/bloom-3b": {
        "batch_size": [1, 4],
        "seq_length": [16, 32],
        "worker": 1,
        "stream": [True],
    },
    "t5-large": {
        "batch_size": [1],
        "seq_length": [32],
        "worker": 1,
        "stream": [True],
    },
    "llama3-tiny-random-lora": {
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
    "mixtral-8x7b": {
        "batch_size": [4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "mistral-7b-rb": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
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
    },
    "tiny-llama-rb": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama-3-1-8b-instruct-vllm-nxdi": {
        "batch_size": [1],
        "seq_length": [256],
    },
    "llama-3-2-1b-instruct-vllm-nxdi-aot": {
        "batch_size": [1],
        "seq_length": [128],
    }
}

transformers_neuronx_neo_model_spec = {
    "llama-3.1-8b-rb": {
        "seq_length": [1024],
        "batch_size": [1, 4],
        "tokenizer": "NousResearch/Meta-Llama-3.1-8B"
    },
    "mixtral-random-tiny": {
        "workers": 1,
        "seq_length": [512],
        "batch_size": [2]
    },
    "codellama-7b-instruct": {
        "workers": 1,
        "seq_length": [256],
        "batch_size": [4]
    },
    "mistral-7b": {
        "workers": 1,
        "seq_length": [512],
        "batch_size": [2]
    },
    "llama-3.1-8b": {
        "workers": 1,
        "seq_length": [128],
        "batch_size": [1]
    },
    "llama-2-tiny-speculative": {
        "workers": 1,
        "seq_length": [128],
        "batch_size": [1]
    }
}

lmi_dist_model_spec = {
    "gpt-neox-20b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b"
    },
    "falcon-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-7b"
    },
    "falcon-11b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-11B"
    },
    "flan-t5-xxl": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "google/flan-t5-xxl"
    },
    "gpt2": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "gpt2"
    },
    "mpt-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "mosaicml/mpt-7b"
    },
    "octocoder": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/octocoder"
    },
    "speculative-llama-13b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "starcoder2-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/starcoder2-7b"
    },
    "gemma-7b": {
        "batch_size": [1, 4],
        "seq_length": [256]
    },
    "gemma-2b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama2-13b-gptq": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
        "parameters": {
            "decoder_input_details": True
        },
        "stream": [False],
    },
    "mistral-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "llama3-8b-chunked-prefill": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    },
    "falcon-11b-chunked-prefill": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-11B"
    },
    "llama2-7b-32k": {
        "batch_size": [1, 4],
        "seq_length": [1024],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
        "parameters": {
            "decoder_input_details": True
        },
        "stream": [False],
    },
    "mistral-7b-128k-awq": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "mistral-7b-marlin": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "llama-2-13b-flashinfer": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
    },
    "llama-7b-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["english-alpaca", "portugese-alpaca", "english-alpaca"],
        "tokenizer": "TheBloke/Llama-2-7B-fp16"
    },
    "llama-7b-unmerged-lora-overflow": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": [f"english-alpaca-{i}" for i in range(20)],
        "tokenizer": "TheBloke/Llama-2-7B-fp16"
    },
    "llama2-13b-awq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "unsloth/mistral-7b-instruct-v0.2"
    },
    "mistral-7b-awq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    },
    "mistral-7b-gptq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    },
    "llama3-8b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "unsloth/llama-3-8b-Instruct"
    },
    "gemma-7b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["alpaca", "dante"],
        "tokenizer": "unsloth/gemma-7b"
    },
    "phi2-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["sql", "bunny"],
        "tokenizer": "microsoft/phi-2"
    },
    "llama-2-tiny": {
        "batch_size": [1, 4],
        "seq_length": [256]
    },
    "llama3-8b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-3-8B-fp16"
    },
    "llama-3.1-8b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Hermes-3-Llama-3.1-8B"
    },
    "llama32-3b-multi-worker-tp1-pp1": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama32-3b-multi-worker-tp2-pp1": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama32-3b-multi-worker-tp1-pp2": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama31-8b-pp-only": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama31-8b-tp2-pp2": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama31-8b-tp2-pp2-spec-dec": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "flan-t5-xl": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "tinyllama-input-len-exceeded": {
        "batch_size": [1],
        "seq_length": [25],
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
}

lmi_dist_chat_model_spec = {
    "llama2-7b-chat": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    }
}

vllm_model_spec = {
    "gpt-neox-20b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b"
    },
    "llama2-13b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
        "parameters": {
            "decoder_input_details": True
        },
        "stream": [False],
    },
    "phi-2": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "microsoft/phi-2"
    },
    "llama2-70b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama3-8b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-3-8B-fp16"
    },
    "mixtral-8x7b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "qwen2-7b-fp8": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "Qwen/Qwen-7B",
        "parameters": {
            "decoder_input_details": True
        },
        "stream": [False]
    },
    "llama3-8b-chunked-prefill": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    },
    "falcon-11b-chunked-prefill": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-11B"
    },
    "llama-68m-speculative-medusa": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "JackFram/llama-68m"
    },
    "llama-68m-speculative-eagle": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "JackFram/llama-68m"
    },
    "llama-7b-unmerged-lora": {
        "batch_size": [3],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["english-alpaca", "portugese-alpaca", "english-alpaca"],
        "tokenizer": "TheBloke/Llama-2-7B-fp16"
    },
    "llama-7b-unmerged-lora-overflow": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": [f"english-alpaca-{i}" for i in range(20)],
        "tokenizer": "TheBloke/Llama-2-7B-fp16"
    },
    "llama2-13b-awq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mistral-7b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "unsloth/mistral-7b-instruct-v0.2"
    },
    "mistral-7b-awq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    },
    "mistral-7b-gptq-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["spanish", "german"],
        "tokenizer": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    },
    "llama3-8b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["french", "spanish"],
        "tokenizer": "unsloth/llama-3-8b-Instruct"
    },
    "gemma-7b-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["alpaca", "dante"],
        "tokenizer": "unsloth/gemma-7b"
    },
    "phi2-unmerged-lora": {
        "batch_size": [4],
        "seq_length": [16, 32],
        "worker": 1,
        "adapters": ["sql", "bunny"],
        "tokenizer": "microsoft/phi-2"
    },
    "starcoder2-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/starcoder2-7b"
    },
    "gemma-7b": {
        "batch_size": [1, 4],
        "seq_length": [256]
    },
    "gemma-2b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "tinyllama-input-len-exceeded": {
        "batch_size": [1],
        "seq_length": [25],
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
}

vllm_neo_model_spec = {
    "tiny-llama-fml": {
        "batch_size": [4],
        "seq_length": [32],
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "tiny-llama-lora-fml": {
        "batch_size": [4],
        "seq_length": [32],
        "adapters": ["tarot"],
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "llama-3.1-8b": {
        "batch_size": [1],
        "seq_length": [256],
        "tokenizer": "NousResearch/Meta-Llama-3.1-8B"
    },
    "llama-3.1-8b-bs4": {
        "batch_size": [4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Meta-Llama-3.1-8B"
    }
}

vllm_chat_model_spec = {
    "llama2-7b-chat": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    },
    "mistral-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16",
    },
    "deepseek-r1-distill-qwen-1-5b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "enable_reasoning": True,
        "tokenizer": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    },
    "llama-3-1-8b-instruct": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "unsloth/Meta-Llama-3.1-8B-Instruct",
    }
}

vllm_tool_model_spec = {
    "llama3-1-8b-instruct-tool": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "unsloth/Meta-Llama-3.1-8B-Instruct"
    },
    "mistral-7b-instruct-v03-tool": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "unsloth/mistral-7b-instruct-v0.3"
    },
}

lmi_dist_aiccl_model_spec = {
    "llama-2-70b-aiccl": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "codellama-34b-aiccl": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "codellama/CodeLlama-34b-hf"
    },
    "falcon-40b-aiccl": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-40b"
    },
    "mixtral-8x7b-aiccl": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
}

trtllm_model_spec = {
    "llama2-13b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "llama2-7b-smoothquant": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "internlm-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "internlm/internlm-7b"
    },
    "baichuan2-13b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "chatglm3-6b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "mistral-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k"
    },
    "gpt-j-6b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-j-6b"
    },
    "qwen-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "Qwen/Qwen-7B"
    },
    "gpt2": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "gpt2"
    },
    "santacoder": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "bigcode/santacoder"
    },
    "llama2-70b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16"
    },
    "mixtral-8x7b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    },
    "flan-t5-xl": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "google/flan-t5-xl",
    },
    "llama-3-1-8b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Meta-Llama-3.1-8B",
    },
}

trtllm_chat_model_spec = {
    "llama2-7b-chat": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-Chat-fp16"
    }
}

trtllm_neo_model_spec = {
    "llama3-8b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Meta-Llama-3-8B"
    },
    "llama3-70b": {
        "batch_size": [1, 8],
        "seq_length": [256],
        "tokenizer": "NousResearch/Meta-Llama-3-70B"
    }
}

no_code_rolling_batch_spec = {
    "llama-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-7B-fp16",
    },
    "llama-13b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-13B-fp16",
    },
    "gemma-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "gemma-2b": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "mistral-7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
    },
    "gpt-neox": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "EleutherAI/gpt-neox-20b",
    },
    "phi-2": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "microsoft/phi-2",
    },
    "baichuan-13b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "baichuan-inc/Baichuan2-13B-Base",
    },
    "qwen-1.5-14b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "Qwen/Qwen1.5-14B",
    },
    "starcoder": {
        "batch_size": [1, 4],
        "seq_length": [256],
    },
    "llama-70b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "TheBloke/Llama-2-70B-fp16",
    },
    "codellama": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "codellama/CodeLlama-34b-hf",
    },
    "mixtral-8x7b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    },
    "falcon-40b": {
        "batch_size": [1, 4],
        "seq_length": [256],
        "tokenizer": "tiiuae/falcon-40b",
    },
    "dbrx": {
        "batch_size": [1, 4],
        "seq_length": [256],
    }
}

correctness_model_spec = {
    "trtllm-codestral-22b": {
        "batch_size": [41],
        "seq_length": [512],
        "num_run": 4,
        "tokenizer": "bullerwins/Codestral-22B-v0.1-hf",
        "dataset": "humaneval",
        "score": 0.04,
        "parameters": {
            "return_full_text": True
        }
    },
    "lmi-dist-codestral-22b": {
        "batch_size": [41],
        "seq_length": [512],
        "num_run": 4,
        "tokenizer": "bullerwins/Codestral-22B-v0.1-hf",
        "dataset": "humaneval",
        "score": 0.5,
        "parameters": {
            "return_full_text": True
        }
    },
    "neuronx-codestral-22b": {
        "batch_size": [41],
        "seq_length": [512],
        "num_run": 4,
        "tokenizer": "bullerwins/Codestral-22B-v0.1-hf",
        "dataset": "humaneval",
        "score": 0.01
    },
    "trtllm-llama3-8b": {
        "batch_size": [213],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "TheBloke/Llama-2-7B-fp16",
        "dataset": "mmlu",
        "score": 0.6
    },
    "lmi-dist-llama3-1-8b": {
        "batch_size": [213],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "TheBloke/Llama-2-7B-fp16",
        "dataset": "mmlu",
        "score": 0.6
    },
    "neuronx-llama3-2-1b": {
        "batch_size": [32],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "NousResearch/Llama-3.2-1B",
        "dataset": "mmlu",
        "score": 0.45
    },
    "trtllm-meta-llama3-8b-fp8": {
        "batch_size": [213],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "TheBloke/Llama-2-7B-fp16",
        "dataset": "mmlu",
        "score": 0.6
    },
    "trtllm-mistral-7b-instruct-v0.3": {
        "batch_size": [213],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
        "dataset": "mmlu",
        "score": 0.59
    },
    "trtllm-mistral-7b-instruct-v0.3-fp8": {
        "batch_size": [213],
        "seq_length": [1],
        "num_run": 66,
        "tokenizer": "amazon/MegaBeam-Mistral-7B-300k",
        "dataset": "mmlu",
        "score": 0.59
    }
}

multi_modal_spec = {
    "llava_v1.6-mistral": {
        "batch_size": [1, 4],
        "tokenizer": "llava-hf/llava-v1.6-mistral-7b-hf"
    },
    "paligemma-3b-mix-448": {
        "batch_size": [1]
    },
    "phi-3-vision-128k-instruct": {
        "batch_size": [1, 4],
        "tokenizer": "microsoft/Phi-3-vision-128k-instruct"
    },
    "pixtral-12b": {
        "batch_size": [1, 4],
    },
    "llama32-11b-multimodal": {
        "batch_size": [1],
    },
}

text_embedding_model_spec = {
    "bge-base-rust": {
        "batch_size": [1, 8],
    },
    "e5-base-v2-rust": {
        "batch_size": [1, 8],
    },
    "sentence-camembert-large-rust": {
        "batch_size": [1, 8],
    },
    "roberta-base-rust": {
        "batch_size": [1, 8],
    },
    "msmarco-distilbert-base-v4-rust": {
        "batch_size": [1, 8],
    },
    "bge-reranker-rust": {
        "batch_size": [1, 8],
        "reranking": True,
    },
    "e5-mistral-7b-rust": {
        "batch_size": [1, 8],
    },
    "gte-qwen2-7b-rust": {
        "batch_size": [1, 8],
    },
    "gte-large-rust": {
        "batch_size": [1, 8],
    },
    "bge-multilingual-gemma2-rust": {
        "batch_size": [1, 8],
    },
    "bge-base-onnx": {
        "batch_size": [1, 8],
    }
}

handler_performance_model_spec = {
    "tiny-llama-model": {
        "batch_size": [1, 512],
        "seq_length": [256],
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
}


def add_file_handler_to_logger(file_path: str):
    handler = logging.FileHandler(file_path, mode='w')
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    return handler


def remove_file_handler_from_logger(handler):
    if handler:
        LOGGER.removeHandler(handler)
        handler.close()


def modelspec_checker(model: str, model_spec: dict):
    if model not in model_spec:
        msg = f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        LOGGER.error(msg)
        raise ValueError(msg)


def check_worker_number(desired):
    model_name = get_model_name()
    endpoint = f"http://127.0.0.1:8080/models/{model_name}"
    res = requests.get(endpoint).json()
    if desired == len(res[0]["models"][0]["workerGroups"]):
        return
    elif desired == len(res[0]["models"][0]["workerGroups"][0]["workers"]):
        return
    else:
        msg = f"Worker number does not meet requirements! {res}"
        LOGGER.error(msg)
        raise AssertionError(msg)


def validate_correctness(type, tasks, expected):
    from llm.correctness.execution import check_correctness

    inputs = []
    outputs = []
    output_dir = os.path.join(os.path.curdir, "outputs")
    for file in os.listdir(output_dir):
        with open(os.path.join(output_dir, file), 'r+') as f:
            for line in f:
                if line and not line == '\n':
                    k, v = line.split(": ", 1)
                    if k == "input":
                        inputs.append(json.loads(v)["inputs"])
                    elif k == "output":
                        try:
                            outputs.append(json.loads(v)["generated_text"])
                        except JSONDecodeError:
                            # delete the last input
                            del inputs[-1]
                            LOGGER.error(f"Failed to read output: {v}")

    if len(outputs) == 0:
        raise RuntimeError(f"No output found in {output_dir}")

    total_pass = 0
    if type == "humaneval":
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = []
            for i, out in enumerate(outputs):
                task = tasks[inputs[i]]
                args = (task, out, TIMEOUT)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                if result['passed']:
                    total_pass += 1
    elif type == "mmlu":
        for i, out in enumerate(outputs):
            task = tasks[inputs[i]]
            if out.strip() == task["answer"]:
                total_pass += 1

    score = total_pass / len(outputs)
    LOGGER.info(
        f'Correctness: {score}, total_pass: {total_pass}, outputs: {len(outputs)}'
    )
    assert score >= expected


def send_json(data):
    headers = {'content-type': 'application/json'}
    endpoint = f"http://127.0.0.1:8080/invocations"
    resp = requests.post(endpoint, headers=headers, json=data)

    if resp.status_code >= 300:
        LOGGER.exception(f"HTTP error: {resp.content}")
        raise ValueError(
            f"Failed to send request to model server. Status code: {resp.status_code}"
        )
    return resp


def find_awscurl():
    command = "./awscurl -h"
    try:
        sp.check_output(command, shell=True)
    except sp.CalledProcessError:
        LOGGER.info("Downloading awscurl...")
        command = "wget https://publish.djl.ai/awscurl/awscurl && chmod +x awscurl"
        sp.call(command, shell=True)


def awscurl_run(data,
                tokenizer,
                concurrency,
                num_run=5,
                dataset=False,
                output=False,
                json_results=False,
                random_delay=False,
                jsonquery=None):
    find_awscurl()
    headers = "'Content-type: application/json'"
    endpoint = f"http://127.0.0.1:8080/invocations"
    if dataset:
        dataset_dir = os.path.join(os.path.curdir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        for i, d in enumerate(data):
            with open(os.path.join(dataset_dir, f"prompt{i}.txt"), "w") as f:
                f.write(json.dumps(d))
        command_data = f"--dataset {dataset_dir}"
    else:
        json_data = json.dumps(data)
        command_data = f"-d '{json_data}'"

    json_output = ""
    if json_results:
        json_output = "--json-path benchmark.json"

    delay = ""
    if random_delay:
        delay = '--delay "rand(0,1000)"'

    jq = ""
    if jsonquery is not None:
        jq = f'-j "{jsonquery}"'

    command = (f"./awscurl -c {concurrency} "
               f"-N {num_run} -X POST {endpoint} --connect-timeout 300 "
               f"-H {headers} {command_data} {delay} {json_output} {jq} -P -t")
    if tokenizer:
        command = f"TOKENIZER={tokenizer} {command}"
    if output:
        output_dir = os.path.join(os.path.curdir, "outputs")
        shutil.rmtree(output_dir, ignore_errors=True)
        os.mkdir(output_dir)
        output_path = os.path.join(output_dir, "output")
        command = f"{command} -o {output_path}"
    LOGGER.info(f"Running command {command}")
    res = sp.run(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    if "error rate: 100" in res.stdout.decode("utf-8"):
        raise ValueError("Found error result in awscurl_run")
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
        LOGGER.exception(f"HTTP error: {resp}")
        raise ValueError("Failed to send request to model server")
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
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "What is deep learning?"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "Who won the world series in 2020?"
        }, {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020."
        }, {
            "role": "user",
            "content": "Where was it played?"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "How do I build a car from cardboard and paper clips?"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "Hello!"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "Who are you?"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "Hello world!"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role": "user",
            "content": "What is the weather like in Brooklyn, New York?"
        }],
        [{
            "role": "user",
            "content": "hello, can you help me?"
        }, {
            "role": "assistant",
            "content": "Hi, what can i help you with today?"
        }, {
            "role":
            "user",
            "content":
            "What's the weather like the next 3 days in San Francisco, CA?"
        }],
    ]

    if batch_size > len(messages):
        # dynamically extend to support larger bs by repetition
        messages *= math.ceil(batch_size / len(messages))
    return messages[:batch_size]


def batch_generation_pair(batch_size):
    data = [{
        "key": "what is panda?",
        "value": "hi"
    }, {
        "key":
        "what is panda?",
        "value":
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    }, {
        "key":
        "What is Deep Learning?",
        "value":
        "Deep learning is a subset of machine learning that utilizes multi-layered neural networks to learn from large amounts of data and perform complex tasks such as image recognition and natural language processing."
    }, {
        "key": "What is Deep Learning?",
        "value": "Deep learning is not"
    }]

    if batch_size > len(data):
        # dynamically extend to support larger bs by repetition
        data *= math.ceil(batch_size / len(data))
    return data[:batch_size]


def batch_generation_tool(batch_size):
    data = [{
        "messages": [{
            "role": "user",
            "content": "Hi! How are you doing today?"
        }, {
            "role": "assistant",
            "content": "I'm doing well! How can I help you?"
        }, {
            "role":
            "user",
            "content":
            "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type":
                            "string",
                            "description":
                            "The city to find the weather for, e.g. 'San Francisco'"
                        },
                        "state": {
                            "type":
                            "string",
                            "description":
                            "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["city", "state", "unit"]
                }
            }
        }],
        "tool_choice":
        "auto"
    }, {
        "messages": [{
            "role": "user",
            "content": "Hi! How are you doing today?"
        }, {
            "role": "assistant",
            "content": "I'm doing well! How can I help you?"
        }, {
            "role":
            "user",
            "content":
            "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type":
                            "string",
                            "description":
                            "The city to find the weather for, e.g. 'San Francisco'"
                        },
                        "state": {
                            "type":
                            "string",
                            "description":
                            "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["city", "state", "unit"]
                }
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_current_weather"
            }
        },
    }]

    if batch_size > len(data):
        # dynamically extend to support larger bs by repetition
        data *= math.ceil(batch_size / len(data))
    return data[:batch_size]


def batch_generation_reasoning(batch_size):
    messages = [
        [{
            "role": "user",
            "content": "9.11 and 9.8, which is greater?"
        }],
        [{
            "role": "user",
            "content": "How many Rs are there in the word 'strawberry'?"
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


def load_dataset(dataset):
    res = {}
    if dataset == "humaneval":
        url = "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/humaneval.jsonl"
        key = "prompt"
    elif dataset == "mmlu":
        url = "https://djl-ai.s3.amazonaws.com/resources/benchmark/datasets/mmlu_djlserving.jsonl"
        key = "inputs"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for line in urllib.request.urlopen(url):
        data = json.loads(line.decode('utf-8'))
        res[data[key]] = data
    return res


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
            msg = f"Logging metrics requires the following arguments: {required_args}"
            LOGGER.error(msg)
            raise ValueError(msg)

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
            for item in message.split('\n'):
                try:
                    if len(item) > 0:
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
            LOGGER.info(
                f"Skipping content check given non-supported content type {res.headers['content-type']}"
            )


def test_handler_rolling_batch(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    # dryrun phase
    req = {"inputs": batch_generation(1)[0]}
    seq_length = spec["seq_length"][0]
    params = {"do_sample": True, "max_new_tokens": seq_length, "details": True}
    req["parameters"] = params
    if "parameters" in spec:
        req["parameters"].update(spec["parameters"])
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]

    for stream in stream_values:
        req["stream"] = stream
        LOGGER.info(f"req {req}")
        res = send_json(req)
        message = res.content.decode("utf-8")
        LOGGER.info(f"res: {message}")
        response_checker(res, message)

    # awscurl little benchmark phase
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            for stream in stream_values:
                req["stream"] = stream
                LOGGER.info(
                    f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
                )
                req["parameters"]["max_new_tokens"] = seq_length
                awscurl_run(req, spec.get("tokenizer", None), batch_size)


def test_handler_adapters(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    # dryrun phase
    reqs = []
    inputs = batch_generation(len(spec.get("adapters")))
    for i, adapter in enumerate(spec.get("adapters")):
        req = {"inputs": inputs[i]}
        seq_length = spec["seq_length"][0]
        params = {
            "do_sample": True,
            "max_new_tokens": seq_length,
            "details": True
        }
        req["parameters"] = params
        req["adapters"] = adapter
        reqs.append(req)
    for req in reqs:
        for stream in stream_values:
            req["stream"] = stream
            LOGGER.info(f"req: {req}")
            res = send_json(req)
            message = res.content.decode("utf-8")
            LOGGER.info(f"res: {message}")
            response_checker(res, message)
    # awscurl little benchmark phase
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            for stream in stream_values:
                LOGGER.info(
                    f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
                )
                for req in reqs:
                    req["parameters"]["max_new_tokens"] = seq_length
                    req["stream"] = stream
                awscurl_run(reqs,
                            spec.get("tokenizer", None),
                            batch_size,
                            dataset=True)
    # Test removing and querying invalid/removed adapter
    del_adapter = spec.get("adapters")[0]
    res = requests.delete(
        f"http://127.0.0.1:8080/models/test/adapters/{del_adapter}")
    LOGGER.info(f"del adapter {res}")
    headers = {'content-type': 'application/json'}
    endpoint = f"http://127.0.0.1:8080/invocations"
    res = requests.post(endpoint, headers=headers,
                        json=reqs[0]).content.decode("utf-8")
    LOGGER.info(f"call deleted adapter {res}")
    assert json.loads(res).get(
        "code"
    ) == FAILED_DEPENDENCY_CODE, "Calling deleted adapter should not work with new adapters"

    if len(reqs) > 1:
        res = requests.post(endpoint, headers=headers,
                            json=reqs[1]).content.decode("utf-8")
        LOGGER.info(f"call valid adapter after deletion {res}")
        final_json = json.loads(res.splitlines()[-1])
        if final_json.get("details", {}).get("finish_reason",
                                             "error") == "error":
            msg = f"Deleting adapter should not break inference for remaining adapters"
            LOGGER.error(msg)
            raise RuntimeError(msg)


def test_handler_rolling_batch_chat(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    # dryrun phase
    if spec.get("enable_reasoning", False):
        req = {"messages": batch_generation_reasoning(1)[0]}
    else:
        req = {"messages": batch_generation_chat(1)[0]}
    req["max_tokens"] = spec["seq_length"][0]
    req["logprobs"] = True
    req["top_logprobs"] = 1
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]

    for stream in stream_values:
        req["stream"] = stream
        LOGGER.info(f"req {req}")
        res = send_json(req)
        LOGGER.info(f"res: {res.content}")
        # awscurl little benchmark phase
        for i, batch_size in enumerate(spec["batch_size"]):
            for seq_length in spec["seq_length"]:
                LOGGER.info(
                    f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
                )
                req["max_tokens"] = seq_length
                awscurl_run(req, spec.get("tokenizer", None), batch_size)

def test_handler_trtllm_rolling_batch_chat(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    # dryrun phase
    if spec.get("enable_reasoning", False):
        req = {"messages": batch_generation_reasoning(1)[0]}
    else:
        req = {"messages": batch_generation_chat(1)[0]}
    req["max_tokens"] = spec["seq_length"][0]
    req["logprobs"] = True
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]

    for stream in stream_values:
        req["stream"] = stream
        LOGGER.info(f"req {req}")
        res = send_json(req)
        LOGGER.info(f"res: {res.content}")
        # awscurl little benchmark phase
        for i, batch_size in enumerate(spec["batch_size"]):
            for seq_length in spec["seq_length"]:
                LOGGER.info(
                    f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
                )
                req["max_tokens"] = seq_length
                awscurl_run(req, spec.get("tokenizer", None), batch_size)

def test_handler_rolling_batch_tool(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    # dryrun phase
    req = batch_generation_tool(1)[0]
    req["max_tokens"] = spec["seq_length"][0]
    req["logprobs"] = True
    req["top_logprobs"] = 1
    if "adapters" in spec:
        req["adapters"] = spec.get("adapters")[0]

    for stream in stream_values:
        req["stream"] = stream
        LOGGER.info(f"req {req}")
        res = send_json(req)
        LOGGER.info(f"res: {res.content}")
        # awscurl little benchmark phase
        for i, batch_size in enumerate(spec["batch_size"]):
            for seq_length in spec["seq_length"]:
                LOGGER.info(
                    f"Little benchmark: concurrency {batch_size} seq_len {seq_length}"
                )
                req["max_tokens"] = seq_length
                awscurl_run(req, spec.get("tokenizer", None), batch_size)


def test_handler(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    stream_values = spec.get("stream", [False, True])
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            for stream in stream_values:
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
                req["stream"] = stream
                LOGGER.info(f"req {req}")
                res = send_json(req)
                if stream:
                    LOGGER.info(f"res: {res.content}")
                    result = res.content.decode().split("\n")[:-1]
                    assert len(
                        result
                    ) <= seq_length, "generated more tokens than max_new_tokens"
                else:
                    res = res.json()
                    LOGGER.info(f"res {res}")
                    if isinstance(res, list):
                        result = [item['generated_text'] for item in res]
                        assert len(result) == batch_size
                    elif isinstance(res, dict):
                        assert 1 == batch_size
                if "tokenizer" in spec:
                    awscurl_run(req, spec.get("tokenizer"), batch_size)


def log_awscurl_benchmark(metric_name: str,
                          benchmark_name: str = "benchmark.json") -> None:
    with open(benchmark_name, "r") as f:
        raw_metrics = json.load(f)
        metrics = list()
        metrics.append({
            "MetricName": f"{metric_name}_p50Latency",
            "Unit": "Milliseconds",
            "Value": raw_metrics["p50Latency"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_p90Latency",
            "Unit": "Milliseconds",
            "Value": raw_metrics["p90Latency"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_p50TimeToFirstByte",
            "Unit": "Milliseconds",
            "Value": raw_metrics["p50TimeToFirstByte"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_p90TimeToFirstByte",
            "Unit": "Milliseconds",
            "Value": raw_metrics["p90TimeToFirstByte"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_tokenThroughput",
            "Unit": "Count/Second",
            "Value": raw_metrics["tokenThroughput"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_tps",
            "Unit": "Count/Second",
            "Value": raw_metrics["tps"]
        })
        metrics.append({
            "MetricName": f"{metric_name}_tokenPerRequest",
            "Unit": "Count",
            "Value": raw_metrics["tokenPerRequest"]
        })
        LOGGER.info(f"{metric_name}")
        LOGGER.info(f"raw metrics: {raw_metrics}")
        command = f'aws cloudwatch put-metric-data --namespace "serving_handler" ' \
                  f'--region "us-east-1" --metric-data \'{json.dumps(metrics)}\''
        LOGGER.info(command)
        sp.call(command, shell=True)


def run_rb_handler_performance(benchmark_name,
                               model_spec,
                               req,
                               jsonquery=None):
    for batch_size in model_spec["batch_size"]:
        metric_name = f"{benchmark_name}-batch-{batch_size:03}"
        num_run = max(
            100 // batch_size,
            10)  # minimum total runs is 100, minimum runs per request is 10
        awscurl_run(req,
                    model_spec.get("tokenizer", None),
                    batch_size,
                    num_run=num_run,
                    json_results=True,
                    random_delay=True,
                    jsonquery=jsonquery)
        log_awscurl_benchmark(metric_name)


def test_handler_performance(benchmark_name, model_spec):
    modelspec_checker("tiny-llama-model", model_spec)
    spec = model_spec["tiny-llama-model"]

    inputs_request = {"inputs": batch_generation(1)[0]}
    inputs_request["max_new_tokens"] = spec["seq_length"][0]
    LOGGER.info(f"{benchmark_name} req {inputs_request}")

    run_rb_handler_performance(f"{benchmark_name}-handler", spec,
                               inputs_request)

    chat_request = {"messages": batch_generation_chat(1)[0]}
    chat_request["max_tokens"] = spec["seq_length"][0]
    LOGGER.info(f"{benchmark_name}-chat req {chat_request}")

    run_rb_handler_performance(f"{benchmark_name}-handler-chat",
                               spec,
                               chat_request,
                               jsonquery="choices/message/content")


def test_performance():
    response_times = []
    for i in range(args.count):
        req = {"inputs": batch_generation(args.batch_size)}
        params = {"max_new_tokens": args.out_tokens}
        req["parameters"] = params
        LOGGER.info(f"req: {req}")
        start = datetime.now()
        res = send_json(req)
        delta = (datetime.now() - start).total_seconds() * 1000
        response_times.append(delta)
        res = res.json()
        LOGGER.info(f"res: {res}")
    log_metrics(response_times)


def test_neuron_sd_handler(model, model_spec):
    from PIL import Image
    modelspec_checker(model, model_spec)
    spec = neuron_sd_model_spec[model]
    for step in spec["num_inference_steps"]:
        req = {"prompt": "A bird and cat flying through space"}
        params = {"num_inference_steps": step}
        req["parameters"] = params
        LOGGER.info(f"req: {req}")
        res = send_json(req)
        try:
            Image.open(BytesIO(res.content)).convert("RGB")
        except Exception as e:
            raise IOError("failed to deserialize image from response", e)


def test_transformers_neuronx_handler(model, model_spec):
    modelspec_checker(model, model_spec)
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for batch_size in spec["batch_size"]:
        inputs = batch_generation(batch_size)
        if batch_size == 1:
            # for rolling batch, inputs should be a str not list.
            # i.e, client side batching is not enabled when rolling batch is enabled.
            # if batch_size is just 1, then we assume it is for rolling batch here.
            inputs = inputs[0]
        for seq_length in spec["seq_length"]:
            req = {"inputs": inputs}
            params = {"max_length": seq_length}
            if "use_sample" in spec:
                params["use_sample"] = True
            req["parameters"] = params
            LOGGER.info(f"req {req}")
            res = send_json(req)
            if spec.get("stream_output", False):
                LOGGER.info(f"res: {res.content}")
            else:
                res = res.json()
                LOGGER.info(f"res {res}")
                result = res
                assert len(result) == batch_size


def test_correctness(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[model]
    score = float(spec.get("score", 0.5))
    parameters = spec.get("parameters", {})
    num_run = int(spec.get("num_run", 5))
    dataset = spec.get("dataset", "mmlu")
    data = load_dataset(dataset)

    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            LOGGER.info(
                f"Correctness testing: concurrency {batch_size} seq_len {seq_length}"
            )
            parameters["max_new_tokens"] = seq_length
            reqs = [{
                "inputs": prompt,
                "parameters": parameters
            } for prompt in data.keys()]
            awscurl_run(reqs,
                        spec.get("tokenizer", None),
                        batch_size,
                        num_run=num_run,
                        dataset=True,
                        output=True)
            validate_correctness(dataset, data, score)


def get_multimodal_prompt(batch_size):
    image_urls = [{
        "type": "image_url",
        "image_url": {
            "url": "https://resources.djl.ai/images/dog_bike_car.jpg",
        }
    }, {
        "type": "image_url",
        "image_url": {
            "url": "https://resources.djl.ai/images/kitten.jpg",
        }
    }, {
        "type": "image_url",
        "image_url": {
            "url": "https://resources.djl.ai/images/kitten_small.jpg",
        }
    }, {
        "type": "image_url",
        "image_url": {
            "url": "https://resources.djl.ai/images/truck.jpg",
        }
    }]

    if batch_size > len(image_urls):
        # dynamically extend to support larger bs by repetition
        image_urls *= math.ceil(batch_size / len(image_urls))

    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is this an image of?",
        }, *image_urls[:batch_size]]
    }]
    return {
        "messages": messages,
        "temperature": 0.9,
        "top_p": 0.6,
        "max_tokens": 512,
    }


def test_multimodal(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not currently supported {list(model_spec.keys())}")
    spec = model_spec[model]
    for i, batch_size in enumerate(spec["batch_size"]):
        req = get_multimodal_prompt(batch_size)
        logging.info(f"req {req}")
        res = send_json(req).json()
        logging.info(f"res: {res}")
        # awscurl little benchmark phase
        awscurl_run(req,
                    spec.get("tokenizer", None),
                    batch_size,
                    num_run=5,
                    output=True)


def test_text_embedding_model(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    reranking = spec.get("reranking", False)
    for i, batch_size in enumerate(spec["batch_size"]):
        if reranking:
            req = batch_generation_pair(batch_size)
        else:
            req = {"inputs": batch_generation(batch_size)}
        logging.info(f"req {req}")
        res = send_json(req).json()
        assert len(res) == batch_size

        # awscurl little benchmark phase
        logging.info(f"Little benchmark: concurrency {batch_size}")
        awscurl_run(req, spec.get("tokenizer"), batch_size)


def run(raw_args):
    parser = argparse.ArgumentParser(description="Build the LLM configs")
    parser.add_argument("handler", help="the handler used in the model")
    parser.add_argument("model", help="The name of model")
    parser.add_argument("--engine",
                        required=False,
                        type=str,
                        choices=["mpi", "huggingface"],
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
    elif args.handler == "transformers_neuronx_neo":
        test_transformers_neuronx_handler(args.model,
                                          transformers_neuronx_neo_model_spec)
    elif args.handler == "transformers_neuronx_neo_rolling_batch":
        test_handler_rolling_batch(args.model,
                                   transformers_neuronx_neo_model_spec)
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
    elif args.handler == "vllm_tool":
        test_handler_rolling_batch_tool(args.model, vllm_tool_model_spec)
    elif args.handler == "vllm_neo":
        test_handler_rolling_batch(args.model, vllm_neo_model_spec)
    elif args.handler == "handler_performance":
        test_handler_performance(args.model, handler_performance_model_spec)
    elif args.handler == "performance":
        test_performance()
    elif args.handler == "lmi_dist_aiccl":
        test_handler_rolling_batch(args.model, lmi_dist_aiccl_model_spec)
    elif args.handler == "trtllm":
        test_handler_rolling_batch(args.model, trtllm_model_spec)
    elif args.handler == "trtllm_chat":
        test_handler_trtllm_rolling_batch_chat(args.model, trtllm_chat_model_spec)
    elif args.handler == "trtllm_neo":
        test_handler_rolling_batch(args.model, trtllm_neo_model_spec)
    elif args.handler == "no_code":
        test_handler_rolling_batch(args.model, no_code_rolling_batch_spec)
    elif args.handler == "correctness":
        test_correctness(args.model, correctness_model_spec)
    elif args.handler == "multimodal":
        test_multimodal(args.model, multi_modal_spec)
    elif args.handler == "text_embedding":
        test_text_embedding_model(args.model, text_embedding_model_spec)

    else:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler")


if __name__ == "__main__":
    run(sys.argv[1:])
