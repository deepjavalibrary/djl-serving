import argparse
import os
import shutil

hf_handler_list = {
    "gpt-neo-2.7b": {
        "option.model_id": "EleutherAI/gpt-neo-2.7B",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.device_map": "auto",
        "option.dtype": "fp16"
    },
    "llama-2-7b": {
        "option.model_id": "s3://djl-llm/llama-2-7b-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.device_map": "auto",
        "option.dtype": "fp16"
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.load_in_8bit": "TRUE",
        "option.device_map": "auto"
    },
    "bigscience/bloom-3b": {
        "option.model_id": "s3://djl-llm/bloom-3b/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "option.task": "text-generation",
        "option.device_map": "auto",
        "option.enable_streaming": True,
        "gpu.maxWorkers": 1,
    },
    "t5-large": {
        "option.model_id": "t5-large",
        "option.tensor_parallel_degree": 1,
        "option.device_map": "auto",
        "option.enable_streaming": True,
    },
    "gpt4all-lora": {
        "option.model_id": "s3://djl-llm/gpt4all-lora/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "fp16"
    }
}

performance_test_list = {
    "open-llama-13b-fp16-lmi-dist": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "engine": "MPI",
        "option.model_id": "s3://djl-llm/open-llama-13b/",
        "option.rolling_batch": "lmi-dist",
    },
    "bloom-7b1-fp16-lmi-dist": {
        "engine": "MPI",
        "option.task": "text-generation",
        "option.rolling_batch": "lmi-dist",
    },
    "gpt-neox-20b-fp16-lmi-dist": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "engine": "MPI",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/",
        "option.rolling_batch": "lmi-dist",
    }
}

transformers_neuronx_aot_handler_list = {
    "gpt2": {
        "option.model_id":
        "gpt2",
        "option.batch_size":
        4,
        "option.tensor_parallel_degree":
        2,
        "option.n_positions":
        512,
        "option.dtype":
        "fp16",
        "option.model_loading_timeout":
        600,
        "option.enable_streaming":
        False,
        "option.save_mp_checkpoint_path":
        "/opt/ml/input/data/training/partition-test"
    },
    "gpt2-quantize": {
        "option.model_id":
        "gpt2",
        "option.batch_size":
        4,
        "option.tensor_parallel_degree":
        2,
        "option.n_positions":
        512,
        "option.dtype":
        "fp16",
        "option.model_loading_timeout":
        600,
        "option.quantize":
        "static_int8",
        "option.enable_streaming":
        False,
        "option.save_mp_checkpoint_path":
        "/opt/ml/input/data/training/partition-test"
    },
}

transformers_neuronx_handler_list = {
    "gpt2": {
        "option.model_id": "gpt2",
        "max_dynamic_batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt2-quantize": {
        "option.model_id": "gpt2",
        "batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.quantize": "static_int8"
    },
    "opt-1.3b": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 512,
        "option.dtype": "fp32",
        "option.model_loading_timeout": 2400
    },
    "pythia-2.8b": {
        "option.model_id": "s3://djl-llm/pythia-2.8b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 900
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 256,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 1200
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16",
        "option.n_positions": 512,
        "option.model_loading_timeout": 2400,
    },
    "opt-1.3b-streaming": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "batch_size": 2,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": True,
    },
    "stable-diffusion-2.1-neuron": {
        "option.model_id":
        "s3://djl-llm/stable-diffusion-2-1-neuron-compiled/",
        "option.height": 512,
        "option.width": 512,
        "batch_size": 1,
        "option.num_images_per_prompt": 1,
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.use_stable_diffusion": True
    },
    "stable-diffusion-1.5-neuron": {
        "option.model_id":
        "s3://djl-llm/stable-diffusion-1-5-neuron-compiled/",
        "option.height": 512,
        "option.width": 512,
        "batch_size": 1,
        "option.num_images_per_prompt": 1,
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.use_stable_diffusion": True
    },
    "stable-diffusion-xl-neuron": {
        "option.model_id": "s3://djl-llm/stable-diffusion-xl-neuron-compiled/",
        "option.height": 1024,
        "option.width": 1024,
        "batch_size": 1,
        "option.num_images_per_prompt": 1,
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.use_stable_diffusion": True
    },
    "llama-7b-rb": {
        "option.model_id": "s3://djl-llm/llama-2-7b-split-inf2/split-model/",
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.max_rolling_batch_size": 4,
        "option.rolling_batch": 'auto',
        "option.model_loading_timeout": 2400,
        "option.load_split_model": True,
        "option.output_formatter": "jsonlines"
    },
    "mistral-7b-rb": {
        "option.model_id": "s3://djl-llm/mistral-7b/",
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.max_rolling_batch_size": 4,
        "option.rolling_batch": 'auto',
        "option.model_loading_timeout": 2400,
        "option.output_formatter": "jsonlines"
    },
    "mixtral-8x7b-rb": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.max_rolling_batch_size": 4,
        "option.rolling_batch": 'auto',
        "option.model_loading_timeout": 3600,
        "option.output_formatter": "jsonlines"
    }
}

rolling_batch_model_list = {
    "gpt2": {
        "option.model_id": "gpt2",
        "engine": "Python",
        "option.max_rolling_batch_size": 4,
        "load_on_devices": 0
    },
    "bloom-560m": {
        "option.model_id": "bigscience/bloom-560m",
        "engine": "Python",
        "option.max_rolling_batch_size": 4,
        "load_on_devices": 0
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "engine": "Python",
        "option.max_rolling_batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "load_on_devices": 0
    },
    "llama2-7b-chat-gptq": {
        "option.model_id": "s3://djl-llm/TheBloke-Llama-2-7b-Chat-GPTQ/",
        "engine": "Python",
        "option.max_rolling_batch_size": 4,
        "load_on_device": 0
    }
}

lmi_dist_model_list = {
    "gpt-neox-20b": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "falcon-7b": {
        "option.model_id": "tiiuae/falcon-7b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 1,
        "option.revision": "2f5c3cd4eace6be6c0f12981f377fb35e5bf6ee5",
        "option.max_rolling_batch_size": 4,
        "option.trust_remote_code": True
    },
    "flan-t5-xxl": {
        "option.model_id": "google/flan-t5-xxl",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "gpt2": {
        "option.model_id": "gpt2",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 2
    },
    "mpt-7b": {
        "option.model_id": "mosaicml/mpt-7b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
        "load_on_devices": 0
    },
    "octocoder": {
        "option.model_id": "s3://djl-llm/octocoder",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "speculative-llama-13b": {
        "option.model_id": "TheBloke/Llama-2-13B-fp16",
        "option.speculative_draft_model": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.gpu_memory_utilization": "0.8",
        "option.tensor_parallel_degree": "max",
    },
    "starcoder2-7b": {
        "option.model_id": "s3://djl-llm/bigcode-starcoder2",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "gemma-7b": {
        "option.model_id": "s3://djl-llm/gemma-7b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "llama2-13b-gptq": {
        "option.model_id": "s3://djl-llm/TheBloke-Llama-2-13b-Chat-GPTQ/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "gptq"
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "llama2-7b-32k": {
        "option.model_id": "togethercomputer/LLaMA-2-7B-32K",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 4
    },
    "mistral-7b-128k-awq": {
        "option.model_id": "TheBloke/Yarn-Mistral-7B-128k-AWQ",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "awq"
    },
    "llama-7b-unmerged-lora": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.adapters": "adapters",
        "option.enable_lora": "true",
        "adapter_ids": ["tloen/alpaca-lora-7b", "22h/cabrita-lora-v0-1"],
        "adapter_names": ["english-alpaca", "portugese-alpaca"],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama-7b-unmerged-lora-overflow": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": 1,
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.adapters": "adapters",
        "option.enable_lora": "true",
        "option.max_cpu_loras": 8,
        "adapter_ids": ["tloen/alpaca-lora-7b"] * 20,
        "adapter_names": [f"english-alpaca-{i}" for i in range(20)],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama2-7b-chat": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    }
}

vllm_model_list = {
    "llama2-13b": {
        "option.model_id": "OpenAssistant/llama2-13b-orca-8k-3319",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.tensor_parallel_degree": 4
    },
    "llama2-13b-awq": {
        "option.model_id": "TheBloke/Llama-2-13B-chat-AWQ",
        "option.quantize": "awq",
        "option.tensor_parallel_degree": 4
    },
    "gpt-neox-20b": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "phi-2": {
        "option.model_id": "microsoft/phi-2",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
    },
    "llama2-70b": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 32,
        "option.output_formatter": "jsonlines"
    },
    "mixtral-8x7b": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 32,
        "option.output_formatter": "jsonlines"
    },
    "llama-7b-unmerged-lora": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.adapters": "adapters",
        "option.enable_lora": "true",
        "adapter_ids": ["tloen/alpaca-lora-7b", "22h/cabrita-lora-v0-1"],
        "adapter_names": ["english-alpaca", "portugese-alpaca"],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama-7b-unmerged-lora-overflow": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": 1,
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.adapters": "adapters",
        "option.enable_lora": "true",
        "option.max_cpu_loras": 8,
        "adapter_ids": ["tloen/alpaca-lora-7b"] * 20,
        "adapter_names": [f"english-alpaca-{i}" for i in range(20)],
        "option.gpu_memory_utilization": "0.8",
    },
    "starcoder2-7b": {
        "option.model_id": "s3://djl-llm/bigcode-starcoder2",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "gemma-7b": {
        "option.model_id": "s3://djl-llm/gemma-7b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
        "option.max_model_len": 3280,
    },
    "llama2-7b-chat": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
    }
}

lmi_dist_aiccl_model_list = {
    "llama-2-70b-aiccl": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
    },
    "codellama-34b-aiccl": {
        "option.model_id": "codellama/CodeLlama-34b-hf",
    },
    "falcon-40b-aiccl": {
        "option.model_id": "s3://djl-llm/falcon-40b/",
    },
    "mixtral-8x7b-aiccl": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
    }
}

trtllm_handler_list = {
    "llama2-13b": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
        "option.output_formatter": "jsonlines",
    },
    "falcon-7b": {
        "option.model_id": "s3://djl-llm/triton/0.9.0/falcon-7b-tp1-bs16/",
        "option.tensor_parallel_degree": 1,
        "option.max_input_len": 1024,
        "option.max_output_len": 512,
        "option.max_rolling_batch_size": 16,
        "option.rolling_batch": "auto",
        "option.output_formatter": "jsonlines",
    },
    "llama2-7b-smoothquant": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
        "option.smoothquant_per_token": "True",
        "option.smoothquant_per_channel": "True",
        "option.rolling_batch": "trtllm",
        "option.output_formatter": "jsonlines",
    },
    "internlm-7b": {
        "option.model_id": "internlm/internlm-7b",
        "option.tensor_parallel_degree": 4,
        "option.output_formatter": "jsonlines",
        "option.trust_remote_code": True
    },
    "baichuan2-13b": {
        "option.model_id": "s3://djl-llm/baichuan2-13b/",
        "option.tensor_parallel_degree": 4,
        "option.baichuan_model_version": "v2_13b",
        "option.output_formatter": "jsonlines",
        "option.trust_remote_code": True
    },
    "chatglm3-6b": {
        "option.model_id": "s3://djl-llm/chatglm3-6b/",
        "option.tensor_parallel_degree": 4,
        "option.output_formatter": "jsonlines",
        "option.trust_remote_code": True,
        "option.chatglm_model_version": "chatglm3"
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
        "option.output_formatter": "jsonlines"
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.tensor_parallel_degree": 1,
        "option.max_input_len": 256,
        "option.max_output_len": 256,
        "option.max_rolling_batch_size": 16,
        "option.rolling_batch": "auto",
        "option.output_formatter": "jsonlines"
    },
    "qwen-7b": {
        "option.model_id": "Qwen/Qwen-7B",
        "option.tensor_parallel_degree": 4,
        "option.trust_remote_code": True,
        "option.output_formatter": "jsonlines"
    },
    "gpt2": {
        "option.model_id": "gpt2",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 16,
        "option.trust_remote_code": True,
        "option.max_draft_len": 20,
        "option.output_formatter": "jsonlines"
    },
    "santacoder": {
        "option.model_id": "bigcode/santacoder",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 16,
        "option.trust_remote_code": True,
        "option.gpt_model_version": "santacoder",
        "option.output_formatter": "jsonlines"
    },
    "llama2-70b": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.use_custom_all_reduce": True,
        "option.max_rolling_batch_size": 32,
        "option.output_formatter": "jsonlines"
    },
    "mixtral-8x7b": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.use_custom_all_reduce": False,
        "option.max_rolling_batch_size": 32,
        "option.output_formatter": "jsonlines"
    },
    "flan-t5-xxl": {
        "engine": "MPI",
        "option.model_id": "s3://djl-llm/flan-t5-xxl-trtllm-compiled/v0.8.0/",
        "option.rolling_batch": "disable",
        "option.entryPoint": "djl_python.tensorrt_llm"
    },
    "flan-t5-xl": {
        "option.model_id": "s3://djl-llm/flan-t5-xl/"
    }
}


def write_model_artifacts(properties,
                          requirements=None,
                          adapter_ids=[],
                          adapter_names=[]):
    model_path = "models/test"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "serving.properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key}={value}\n")
    if requirements:
        with open(os.path.join(model_path, "requirements.txt"), "w") as f:
            f.write('\n'.join(requirements) + '\n')

    adapters_path = os.path.abspath(os.path.join(model_path, "adapters"))
    # Download adapters if any
    if adapter_ids:
        os.makedirs(adapters_path, exist_ok=True)
        ## install huggingface_hub in your workflow file to use this
        from huggingface_hub import snapshot_download
        adapter_cache = {}
        for adapter_id, adapter_name in zip(adapter_ids, adapter_names):
            dir = os.path.join(adapters_path, adapter_name)
            if adapter_id in adapter_cache:
                shutil.copytree(adapter_cache[adapter_id], dir)
            else:
                os.makedirs(dir, exist_ok=True)
                snapshot_download(adapter_id,
                                  local_dir_use_symlinks=False,
                                  local_dir=dir)
                adapter_cache[adapter_id] = dir


def build_hf_handler_model(model):
    if model not in hf_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(hf_handler_list.keys())}"
        )
    options = hf_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.huggingface"
    options["option.predict_timeout"] = 240
    options["option.rolling_batch"] = "disable"

    adapter_ids = options.pop("adapter_ids", [])
    adapter_names = options.pop("adapter_names", [])

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names)


def build_performance_model(model):
    if model in performance_test_list.keys():
        options = performance_test_list[model]
    else:
        options = {"option.task": "text-generation", "option.model_id": model}
    options["option.predict_timeout"] = 240
    if args.dtype:
        options["option.dtype"] = args.dtype
    if options.get('option.dtype') is None:
        raise ValueError("Need to provide dtype for performance benchmark")
    options["option.tensor_parallel_degree"] = args.tensor_parallel
    engine = options.get('engine')
    if args.engine:
        engine = args.engine
        options['engine'] = engine
    if engine is None:
        raise ValueError("Need to provide engine for performance benchmark")
    write_model_artifacts(options)


def build_transformers_neuronx_handler_model(model):
    if model not in transformers_neuronx_handler_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(transformers_neuronx_handler_list.keys())}"
        )
    options = transformers_neuronx_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.transformers_neuronx"
    write_model_artifacts(options)


def build_transformers_neuronx_aot_handler_model(model):
    if model not in transformers_neuronx_aot_handler_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(transformers_neuronx_aot_handler_list.keys())}"
        )
    options = transformers_neuronx_aot_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.transformers_neuronx"
    write_model_artifacts(options)


def build_rolling_batch_model(model):
    if model not in rolling_batch_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(rolling_batch_model_list.keys())}"
        )
    options = rolling_batch_model_list[model]
    options["rolling_batch"] = "scheduler"
    write_model_artifacts(options)


def build_lmi_dist_model(model):
    if model not in lmi_dist_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(lmi_dist_model_list.keys())}"
        )
    options = lmi_dist_model_list[model]
    options["engine"] = "MPI"
    options["option.rolling_batch"] = "lmi-dist"
    options["option.output_formatter"] = "jsonlines"

    adapter_ids = options.pop("adapter_ids", [])
    adapter_names = options.pop("adapter_names", [])

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names)


def build_vllm_model(model):
    if model not in vllm_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_model_list.keys())}"
        )
    options = vllm_model_list[model]
    options["engine"] = "Python"
    options["option.rolling_batch"] = "vllm"
    options["option.output_formatter"] = "jsonlines"

    adapter_ids = options.pop("adapter_ids", [])
    adapter_names = options.pop("adapter_names", [])

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names)


def build_lmi_dist_aiccl_model(model):
    if model not in lmi_dist_aiccl_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(lmi_dist_aiccl_model_list.keys())}"
        )
    options = lmi_dist_aiccl_model_list[model]
    options["engine"] = "MPI"
    options["option.task"] = "text-generation"
    options["option.tensor_parallel_degree"] = 8
    options["option.rolling_batch"] = "lmi-dist"
    options["option.output_formatter"] = "jsonlines"
    options["option.max_rolling_batch_size"] = 16
    write_model_artifacts(options)


def build_trtllm_handler_model(model):
    if model not in trtllm_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(trtllm_handler_list.keys())}"
        )
    options = trtllm_handler_list[model]
    # 30 minute waiting for conversion timeout
    options["model_loading_timeout"] = "1800"
    write_model_artifacts(options)


supported_handler = {
    'huggingface': build_hf_handler_model,
    'transformers_neuronx': build_transformers_neuronx_handler_model,
    'transformers_neuronx_aot': build_transformers_neuronx_aot_handler_model,
    'performance': build_performance_model,
    'rolling_batch_scheduler': build_rolling_batch_model,
    'lmi_dist': build_lmi_dist_model,
    'lmi_dist_aiccl': build_lmi_dist_aiccl_model,
    'vllm': build_vllm_model,
    'trtllm': build_trtllm_handler_model,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the LLM configs')
    parser.add_argument('handler', help='the handler used in the model')
    parser.add_argument('model', help='model that works with certain handler')
    parser.add_argument('--engine',
                        required=False,
                        type=str,
                        choices=['deepspeed', 'huggingface'],
                        help='The engine used for inference')
    parser.add_argument('--dtype',
                        required=False,
                        type=str,
                        help='The model data type')
    parser.add_argument('--tensor_parallel',
                        required=False,
                        type=int,
                        help='The model tensor parallel degree')
    global args
    args = parser.parse_args()

    if args.handler not in supported_handler:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}"
        )
    supported_handler[args.handler](args.model)
