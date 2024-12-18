import argparse
import os
import subprocess
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
    "llama3-tiny-random-lora": {
        "option.model_id": "llamafactory/tiny-random-Llama-3-lora",
        "option.tensor_parallel_degree": 4,
        "option.device_map": "auto",
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

transformers_neuronx_handler_list = {
    "gpt2": {
        "option.model_id": "s3://djl-llm/gpt2/",
        "max_dynamic_batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt2-quantize": {
        "option.model_id": "s3://djl-llm/gpt2/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.quantize": "static_int8"
    },
    "opt-1.3b": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp32",
        "option.model_loading_timeout": 2400
    },
    "pythia-2.8b": {
        "option.model_id": "s3://djl-llm/pythia-2.8b/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 900
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 256,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 1200
    },
    "mixtral-8x7b": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.rolling_batch": "disable",
        "batch_size": 4,
        "option.model_loading_timeout": 3600,
    },
    "opt-1.3b-streaming": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "batch_size": 2,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": True,
    },
    "stable-diffusion-2.1-neuron": {
        "option.model_id":
        "s3://djl-llm/optimum/latest/stable-diffusion-2-1-neuron-compiled/",
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
        "s3://djl-llm/optimum/latest/stable-diffusion-1-5-neuron-compiled/",
        "option.height": 512,
        "option.width": 512,
        "batch_size": 1,
        "option.num_images_per_prompt": 1,
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.use_stable_diffusion": True
    },
    "stable-diffusion-xl-neuron": {
        "option.model_id":
        "s3://djl-llm/optimum/latest/stable-diffusion-xl-neuron-compiled/",
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
        "option.model_loading_timeout": 2400,
        "option.load_split_model": True,
    },
    "llama-3-8b-rb-vllm": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.max_rolling_batch_size": 4,
        "option.rolling_batch": 'vllm',
        "option.model_loading_timeout": 2400,
    },
    "tiny-llama-rb-vllm": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 1024,
        "option.max_rolling_batch_size": 4,
        "option.rolling_batch": 'vllm',
        "option.model_loader": 'vllm',
        "option.model_loading_timeout": 1200,
    },
    "mistral-7b-rb": {
        "option.model_id": "s3://djl-llm/mistral-7b-instruct-v02/",
        "option.max_rolling_batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 1024,
        "option.model_loading_timeout": 2400,
    },
    "llama-speculative-rb": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.speculative_draft_model": "s3://djl-llm/llama-2-tiny/",
        "option.speculative_length": 7,
        "option.tensor_parallel_degree": 12,
        "option.max_rolling_batch_size": 1,
        "option.model_loading_timeout": 3600,
    },
    "llama-speculative-compiled-rb": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.compiled_graph_path":
        "s3://djl-llm/inf2-compiled-graphs/llama-2-13b-hf/",
        "option.speculative_draft_model": "s3://djl-llm/llama-2-tiny/",
        "option.draft_model_compiled_path":
        "s3://djl-llm/inf2-compiled-graphs/llama-2-tiny/",
        "option.speculative_length": 4,
        "option.tensor_parallel_degree": 12,
        "option.max_rolling_batch_size": 1,
        "option.model_loading_timeout": 3600,
    },
    "tiny-llama-rb-lcnc": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.rolling_batch": "auto",
        "option.model_loading_timeout": 3600,
    },
    "tiny-llama-rb-aot": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.rolling_batch": "auto",
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 1024,
        "option.max_rolling_batch_size": 4,
        "option.model_loading_timeout": 1200,
    },
    "tiny-llama-rb-aot-quant": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.quantize": "static_int8",
        "option.rolling_batch": "auto",
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 1024,
        "option.max_rolling_batch_size": 4,
        "option.model_loading_timeout": 1200,
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
        "option.max_rolling_batch_size": 4,
    },
    "falcon-11b": {
        "option.model_id": "s3://djl-llm/falcon-11B/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 4,
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
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
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
        "option.max_model_len": 2656,
    },
    "gemma-2b": {
        "option.model_id": "s3://djl-llm/gemma-2b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 256,
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
    # TODO: Adding max_model_len due to changes mem profiling
    #       for RoPE scaling models in vLLM
    "llama2-7b-32k": {
        "option.model_id": "togethercomputer/LLaMA-2-7B-32K",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 4,
        "option.max_model_len": 51888,
    },
    "mistral-7b-128k-awq": {
        "option.model_id": "TheBloke/Yarn-Mistral-7B-128k-AWQ",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 4,
        "option.max_model_len": 32768,
        "option.quantize": "awq"
    },
    "mistral-7b-marlin": {
        "option.model_id": "neuralmagic/OpenHermes-2.5-Mistral-7B-marlin",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "marlin"
    },
    "llama-2-13b-flashinfer": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
    },
    "llama3-8b": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
    },
    "llama3-8b-chunked-prefill": {
        "option.model_id": "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.enable_chunked_prefill": "true",
    },
    "falcon-11b-chunked-prefill": {
        "option.model_id": "s3://djl-llm/falcon-11B/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.enable_chunked_prefill": "true",
    },
    "llama-7b-unmerged-lora": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.enable_lora": "true",
        "option.max_loras": 2,
        "option.max_lora_rank": 16,
        "option.long_lora_scaling_factors": "4.0",
        "option.adapters": "adapters",
        "adapter_ids": ["tloen/alpaca-lora-7b", "22h/cabrita-lora-v0-1"],
        "adapter_names": ["english-alpaca", "portugese-alpaca"],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama-7b-unmerged-lora-overflow": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.enable_lora": "true",
        "option.max_loras": 6,
        "option.max_cpu_loras": 8,
        "option.adapters": "adapters",
        "adapter_ids": ["tloen/alpaca-lora-7b"] * 20,
        "adapter_names": [f"english-alpaca-{i}" for i in range(20)],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama2-7b-chat": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "llama2-13b-awq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/TheBloke-Llama-2-13b-Chat-AWQ/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "awq",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/llama-2-13b-chat-fr",
            "UnderstandLing/llama-2-13b-chat-es"
        ],
        "adapter_names": ["french", "spanish"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-awq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02-awq/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "awq",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.lora_dtype":
        "float16",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-gptq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02-gptq/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "gptq",
        "option.dtype":
        "fp16",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.lora_dtype":
        "float16",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "llama3-8b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Llama-3-8B-Instruct-fr",
            "UnderstandLing/Llama-3-8B-Instruct-es",
        ],
        "adapter_names": ["french", "spanish"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "gemma-7b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/gemma-7b/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        1,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "Chuanming/Alpaca-Gemma-7b-lora",
            "girtcius/gemma-7b-dante-lora",
        ],
        "adapter_names": ["alpaca", "dante"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "phi2-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/phi-2/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        1,
        "option.max_lora_rank":
        128,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "isotr0py/phi-2-test-sql-lora",
            "BAAI/bunny-phi-2-siglip-lora",
        ],
        "adapter_names": ["sql", "bunny"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "llama-2-tiny": {
        "option.model_id": "s3://djl-llm/llama-2-tiny/",
        "option.quantize": "awq",
        "option.tensor_parallel_degree": 4,
        "option.device_map": "auto"
    },
    "llama-3.1-8b": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "llava_v1.6-mistral": {
        "option.model_id": "s3://djl-llm/llava-v1.6-mistral-7b-hf/",
        "option.limit_mm_per_prompt": "image=4",
    },
    "paligemma-3b-mix-448": {
        "option.model_id": "s3://djl-llm/paligemma-3b-mix-448/",
        "option.tensor_parallel_degree": 1,
    },
    "phi-3-vision-128k-instruct": {
        "option.model_id": "s3://djl-llm/phi-3-vision-128k-instruct/",
        "option.limit_mm_per_prompt": "image=4",
        "option.trust_remote_code": True,
        "option.max_model_len": 8192,
    },
    "pixtral-12b": {
        "option.model_id": "s3://djl-llm/pixtral-12b/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.tokenizer_mode": "mistral",
        "option.limit_mm_per_prompt": "image=4",
        "option.entryPoint": "djl_python.huggingface"
    },
    "llama32-11b-multimodal": {
        "option.model_id": "s3://djl-llm/llama-3-2-11b-vision-instruct/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.enforce_eager": True,
    },
    "llama32-3b-multi-worker-tp1-pp1": {
        "option.model_id": "s3://djl-llm/llama-3-2-3b-instruct/",
        "option.tensor_parallel_degree": 1,
        "option.pipeline_parallel_degree": 1,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
    },
    "llama32-3b-multi-worker-tp2-pp1": {
        "option.model_id": "s3://djl-llm/llama-3-2-3b-instruct/",
        "option.tensor_parallel_degree": 2,
        "option.pipeline_parallel_degree": 1,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
    },
    "llama32-3b-multi-worker-tp1-pp2": {
        "option.model_id": "s3://djl-llm/llama-3-2-3b-instruct/",
        "option.tensor_parallel_degree": 1,
        "option.pipeline_parallel_degree": 2,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
    },
    "llama31-8b-pp-only": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-instruct-hf/",
        "option.tensor_parallel_degree": 1,
        "option.pipeline_parallel_degree": 4,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
    },
    "llama31-8b-tp2-pp2": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-instruct-hf/",
        "option.tensor_parallel_degree": 2,
        "option.pipeline_parallel_degree": 2,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
    },
    "llama31-8b-tp2-pp2-spec-dec": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-instruct-hf/",
        "option.tensor_parallel_degree": 2,
        "option.pipeline_parallel_degree": 2,
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.speculative_draft_model":
        "s3://djl-llm/llama-3-2-1b-instruct/",
    },
    "flan-t5-xl": {
        "option.model_id": "s3://djl-llm/flan-t5-xl/",
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
    },
    "mixtral-8x7b": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 32,
    },
    "qwen2-7b-fp8": {
        "option.model_id": "neuralmagic/Qwen2-7B-Instruct-FP8",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "fp8"
    },
    "llama3-8b-chunked-prefill": {
        "option.model_id": "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.enable_chunked_prefill": "true",
    },
    "falcon-11b-chunked-prefill": {
        "option.model_id": "s3://djl-llm/falcon-11B/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.enable_chunked_prefill": "true",
    },
    "llama-68m-speculative-medusa": {
        "option.model_id": "s3://djl-llm/llama-68m/",
        "option.task": "text-generation",
        "option.speculative_model": "abhigoyal/vllm-medusa-llama-68m-random",
        "option.num_speculative_tokens": 4,
        "option.use_v2_block_manager": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "llama-68m-speculative-eagle": {
        "option.model_id": "s3://djl-llm/llama-68m/",
        "option.task": "text-generation",
        "option.speculative_model": "abhigoyal/vllm-eagle-llama-68m-random",
        "option.num_speculative_tokens": 4,
        "option.use_v2_block_manager": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "llama-7b-unmerged-lora": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.enable_lora": "true",
        "option.max_loras": 2,
        "option.max_lora_rank": 16,
        "option.long_lora_scaling_factors": "4.0",
        "option.adapters": "adapters",
        "adapter_ids": ["tloen/alpaca-lora-7b", "22h/cabrita-lora-v0-1"],
        "adapter_names": ["english-alpaca", "portugese-alpaca"],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama-7b-unmerged-lora-overflow": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": "max",
        "option.enable_lora": "true",
        "option.max_loras": 6,
        "option.max_cpu_loras": 8,
        "option.adapters": "adapters",
        "adapter_ids": ["tloen/alpaca-lora-7b"] * 20,
        "adapter_names": [f"english-alpaca-{i}" for i in range(20)],
        "option.gpu_memory_utilization": "0.8",
    },
    "llama2-13b-awq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/TheBloke-Llama-2-13b-Chat-AWQ/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "awq",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/llama-2-13b-chat-fr",
            "UnderstandLing/llama-2-13b-chat-es"
        ],
        "adapter_names": ["french", "spanish"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-awq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02-awq/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "awq",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.lora_dtype":
        "float16",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "mistral-7b-gptq-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/mistral-7b-instruct-v02-gptq/",
        "option.tensor_parallel_degree":
        "max",
        "option.quantize":
        "gptq",
        "option.dtype":
        "fp16",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.lora_dtype":
        "float16",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Mistral-7B-Instruct-v0.2-es",
            "UnderstandLing/Mistral-7B-Instruct-v0.2-de"
        ],
        "adapter_names": ["spanish", "german"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "llama3-8b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        2,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "UnderstandLing/Llama-3-8B-Instruct-fr",
            "UnderstandLing/Llama-3-8B-Instruct-es",
        ],
        "adapter_names": ["french", "spanish"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "gemma-7b-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/gemma-7b/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        1,
        "option.max_lora_rank":
        64,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "Chuanming/Alpaca-Gemma-7b-lora",
            "girtcius/gemma-7b-dante-lora",
        ],
        "adapter_names": ["alpaca", "dante"],
        "option.gpu_memory_utilization":
        "0.8",
    },
    "phi2-unmerged-lora": {
        "option.model_id":
        "s3://djl-llm/phi-2/",
        "option.tensor_parallel_degree":
        "max",
        "option.enable_lora":
        "true",
        "option.max_loras":
        1,
        "option.max_lora_rank":
        128,
        "option.long_lora_scaling_factors":
        "4.0",
        "option.adapters":
        "adapters",
        "adapter_ids": [
            "isotr0py/phi-2-test-sql-lora",
            "BAAI/bunny-phi-2-siglip-lora",
        ],
        "adapter_names": ["sql", "bunny"],
        "option.gpu_memory_utilization":
        "0.8",
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
        "option.max_model_len": 2656,
    },
    "gemma-2b": {
        "option.model_id": "s3://djl-llm/gemma-2b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 256,
    },
    "llama2-7b-chat": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
    },
    "llava_v1.6-mistral": {
        "option.model_id": "s3://djl-llm/llava-v1.6-mistral-7b-hf/",
        "option.limit_mm_per_prompt": "image=4",
    },
    "paligemma-3b-mix-448": {
        "option.model_id": "s3://djl-llm/paligemma-3b-mix-448/",
        "option.tensor_parallel_degree": 1,
    },
    "phi-3-vision-128k-instruct": {
        "option.model_id": "s3://djl-llm/phi-3-vision-128k-instruct/",
        "option.limit_mm_per_prompt": "image=4",
        "option.trust_remote_code": True,
        "option.max_model_len": 8192,
    },
    "pixtral-12b": {
        "option.model_id": "s3://djl-llm/pixtral-12b/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.tokenizer_mode": "mistral",
        "option.limit_mm_per_prompt": "image=4",
        "option.entryPoint": "djl_python.huggingface",
        "option.tensor_parallel_degree": "max"
    },
    "llama32-11b-multimodal": {
        "option.model_id": "s3://djl-llm/llama-3-2-11b-vision-instruct/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.enforce_eager": True,
    }
}

vllm_neo_model_list = {
    "tiny-llama-fml": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.tensor_parallel_degree": 2,
        "option.load_format": "sagemaker_fast_model_loader",
    },
    "tiny-llama-lora-fml": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.tensor_parallel_degree": 2,
        "option.load_format": "sagemaker_fast_model_loader",
        "option.adapters": "adapters",
        "option.enable_lora": "true",
        "option.max_lora_rank": "64",
        "adapter_ids": ["barissglc/tinyllama-tarot-v1"],
        "adapter_names": ["tarot"],
    },
    "llama-3.1-8b": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
    },
    "llama-3.1-8b-awq-options": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": "4",
        "option.max_rolling_batch_size": "4",
        "option.awq_block_size": "256"
    },
    "llama-3.1-8b-fp8-options": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.quantize": "fp8",
        "option.tensor_parallel_degree": "4",
        "option.fp8_activation_scheme": "dynamic"
    }
}

lmi_dist_aiccl_model_list = {
    "llama-2-70b-aiccl": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
    },
    "codellama-34b-aiccl": {
        "option.model_id": "s3://djl-llm/CodeLlama-34b-Instruct-hf/",
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
    },
    "llama2-7b-smoothquant": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
        "option.smoothquant_per_token": "True",
        "option.smoothquant_per_channel": "True",
        "option.rolling_batch": "trtllm",
    },
    "internlm-7b": {
        "option.model_id": "internlm/internlm-7b",
        "option.tensor_parallel_degree": 4,
        "option.trust_remote_code": True
    },
    "baichuan2-13b": {
        "option.model_id": "s3://djl-llm/baichuan2-13b/",
        "option.tensor_parallel_degree": 4,
        "option.baichuan_model_version": "v2_13b",
        "option.trust_remote_code": True
    },
    "chatglm3-6b": {
        "option.model_id": "s3://djl-llm/chatglm3-6b/",
        "option.tensor_parallel_degree": 4,
        "option.trust_remote_code": True,
        "option.chatglm_model_version": "chatglm3"
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.tensor_parallel_degree": 1,
        "option.max_input_len": 256,
        "option.max_output_len": 256,
        "option.max_rolling_batch_size": 16,
        "option.rolling_batch": "auto",
    },
    "qwen-7b": {
        "option.model_id": "Qwen/Qwen-7B",
        "option.tensor_parallel_degree": 4,
        "option.trust_remote_code": True,
    },
    "gpt2": {
        "option.model_id": "gpt2",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 16,
        "option.trust_remote_code": True,
        "option.max_draft_len": 20,
    },
    "santacoder": {
        "option.model_id": "bigcode/santacoder",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 16,
        "option.trust_remote_code": True,
        "option.gpt_model_version": "santacoder",
    },
    "llama2-70b": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.use_custom_all_reduce": True,
        "option.max_rolling_batch_size": 32,
    },
    "mixtral-8x7b": {
        "option.model_id": "s3://djl-llm/mixtral-8x7b/",
        "option.tensor_parallel_degree": 8,
        "option.use_custom_all_reduce": False,
        "option.max_rolling_batch_size": 32,
    },
    "llama2-7b-chat": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.dtype": "fp16",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "flan-t5-xl": {
        "option.model_id": "s3://djl-llm/flan-t5-xl/",
        "option.dtype": "bf16",
        "option.max_rolling_batch_size": 128,
        # This is needed in v12, but we don't know exactly why the default max_utilization does not work
        "option.batch_scheduler_policy": "guaranteed_no_evict",
    },
    "llama-3-1-8b": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": 4,
    },
}

correctness_model_list = {
    "trtllm-codestral-22b": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "bullerwins/Codestral-22B-v0.1-hf",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 41
    },
    "lmi-dist-codestral-22b": {
        "engine": "MPI",
        "option.task": "text-generation",
        "option.model_id": "bullerwins/Codestral-22B-v0.1-hf",
        "option.rolling_batch": "lmi-dist",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 41
    },
    "neuronx-codestral-22b": {
        "engine": "Python",
        "option.entryPoint": "djl_python.transformers_neuronx",
        "option.model_id": "bullerwins/Codestral-22B-v0.1-hf",
        "option.tensor_parallel_degree": 12,
        "option.n_positions": 1024,
        "option.rolling_batch": "auto",
        "option.max_rolling_batch_size": 41,
        "option.model_loading_timeout": 1800
    },
    "trtllm-llama3-8b": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 213
    },
    "lmi-dist-llama3-1-8b": {
        "engine": "MPI",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.rolling_batch": "lmi-dist",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 213
    },
    "neuronx-llama3-1-8b": {
        "engine": "Python",
        "option.entryPoint": "djl_python.transformers_neuronx",
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": 12,
        "option.n_positions": 768,
        "option.rolling_batch": "auto",
        "option.max_rolling_batch_size": 213,
        "option.model_loading_timeout": 1800
    },
    "trtllm-meta-llama3-8b-fp8": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "fp8"
    },
    "trtllm-mistral-7b-instruct-v0.3": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/mistral-7b-instruct-v03/",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4
    },
    "trtllm-mistral-7b-instruct-v0.3-fp8": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/mistral-7b-instruct-v03/",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "fp8"
    }
}

trtllm_neo_list = {
    "llama3-8b-tp1-fp16": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.tensor_parallel_degree": 1,
        "option.rolling_batch": "trtllm",
    },
    "llama3-8b-tp4-awq": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
        "option.quantize": "awq"
    },
    "llama3-8b-tp4-fp8": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
        "option.quantize": "fp8"
    },
    "llama3-8b-tp4-smoothquant": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.rolling_batch": "trtllm",
        "option.quantize": "smoothquant"
    },
    "llama3-70b-tp8-fp16": {
        "option.model_id": "s3://djl-llm/llama-3-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.rolling_batch": "trtllm",
    },
    "llama3-70b-tp8-awq": {
        "option.model_id": "s3://djl-llm/llama-3-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.rolling_batch": "trtllm",
        "option.quantize": "awq"
    },
    "llama3-70b-tp8-fp8": {
        "option.model_id": "s3://djl-llm/llama-3-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.rolling_batch": "trtllm",
        "option.quantize": "fp8"
    },
    "llama3-70b-tp8-smoothquant": {
        "option.model_id": "s3://djl-llm/llama-3-70b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.rolling_batch": "trtllm",
        "option.quantize": "smoothquant"
    }
}

transformers_neuronx_neo_list = {
    "llama-3.1-8b-rb": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.rolling_batch": "auto",
        "option.max_rolling_batch_size": 4,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 3600,
    },
    "mixtral-random-tiny": {
        "option.model_id": "s3://djl-llm/mixtral-random-tiny/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.batch_size": 2,
        "option.model_loading_timeout": 3600,
    },
    "codellama-7b-instruct": {
        "option.model_id": "s3://djl-llm/CodeLlama-7b-Instruct-hf/",
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 256,
        "option.rolling_batch": "disable",
        "option.batch_size": 4,
        "option.model_loading_timeout": 3600,
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b/",
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.rolling_batch": "disable",
        "option.batch_size": 2,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 3600,
    },
    "llama-3.1-8b": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": 8
    },
    "llama-2-tiny-speculative": {
        "option.model_id": "s3://djl-llm/llama-2-tiny/",
        "option.speculative_draft_model": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.tensor_parallel_degree": 2,
        "option.max_rolling_batch_size": 1
    }
}

text_embedding_model_list = {
    "bge-base-rust": {
        "engine": "Rust",
        "option.model_id": "BAAI/bge-base-en-v1.5",
        "batch_size": 8,
    },
    "e5-base-v2-rust": {
        "engine": "Rust",
        "option.model_id": "intfloat/e5-base-v2",
        "pooling": "cls",
        "batch_size": 8,
    },
    "sentence-camembert-large-rust": {
        "engine": "Rust",
        "option.model_id": "dangvantuan/sentence-camembert-large",
        "pooling": "cls",
        "batch_size": 8,
    },
    "roberta-base-rust": {
        "engine": "Rust",
        "option.model_id": "relbert/relbert-roberta-base-nce-conceptnet",
        "pooling": "cls",
        "batch_size": 8,
    },
    "msmarco-distilbert-base-v4-rust": {
        "engine": "Rust",
        "option.model_id": "sentence-transformers/msmarco-distilbert-base-v4",
        "pooling": "cls",
        "batch_size": 8,
    },
    "bge-reranker-rust": {
        "engine": "Rust",
        "option.model_id": "BAAI/bge-reranker-base",
        "reranking": True,
        "batch_size": 8,
    },
    "e5-mistral-7b-rust": {
        "engine": "Rust",
        "option.model_id": "intfloat/e5-mistral-7b-instruct",
        "pooling": "cls",
        "batch_size": 8,
    },
    "gte-qwen2-7b-rust": {
        "engine": "Rust",
        "option.model_id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "pooling": "cls",
        "batch_size": 8,
    },
    "gte-large-rust": {
        "engine": "Rust",
        "option.model_id": "Alibaba-NLP/gte-large-en-v1.5",
        "option.trust_remote_code": "true",
        "pooling": "cls",
        "batch_size": 8,
    },
    "bge-multilingual-gemma2-rust": {
        "engine": "Rust",
        "option.model_id": "BAAI/bge-multilingual-gemma2",
        "pooling": "cls",
        "batch_size": 8,
    },
    "bge-base-onnx": {
        "engine": "OnnxRuntime",
        "option.model_id": "BAAI/bge-base-en-v1.5",
        "batch_size": 8,
    }
}

handler_performance_model_list = {
    "tiny-llama-lmi": {
        "engine": "MPI",
        "option.model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "option.rolling_batch": "lmi-dist",
        "option.max_rolling_batch_size": 512,
    },
    "tiny-llama-vllm": {
        "engine": "Python",
        "option.model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "option.task": "text-generation",
        "option.rolling_batch": "vllm",
        "option.gpu_memory_utilization": "0.9",
        "option.max_rolling_batch_size": 512,
    },
    "tiny-llama-trtllm": {
        "engine": "Python",
        "option.model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "option.rolling_batch": "trtllm",
        "option.max_rolling_batch_size": 512,
    },
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


def create_neo_input_model(properties):
    model_path = "models"
    model_download_path = os.path.join(model_path, "uncompiled")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_download_path, exist_ok=True)
    with open(os.path.join(model_download_path, "serving.properties"),
              "w") as f:
        for key, value in properties.items():
            if key != "option.model_id":
                f.write(f"{key}={value}\n")

    # create Neo files/dirs
    open(os.path.join(model_path, "errors.json"), "w").close()
    os.makedirs(os.path.join(model_path, "cache"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "compiled"), exist_ok=True)

    # Download the model checkpoint from S3 to local path
    model_s3_uri = properties.get("option.model_id")
    if os.path.isfile("/opt/djl/bin/s5cmd"):
        if not model_s3_uri.endswith("*"):
            if model_s3_uri.endswith("/"):
                model_s3_uri += '*'
            else:
                model_s3_uri += '/*'

        cmd = ["/opt/djl/bin/s5cmd", "sync", model_s3_uri, model_download_path]
    else:
        cmd = ["aws", "s3", "sync", model_s3_uri, model_download_path]
    subprocess.check_call(cmd)

    adapter_ids = properties.pop("adapter_ids", [])
    adapter_names = properties.pop("adapter_names", [])
    # Copy Adapters if any
    if adapter_ids:
        print("copying adapter models")
        adapters_path = os.path.join(model_download_path, "adapters")
        os.makedirs(adapters_path, exist_ok=True)
        ## install huggingface_hub in your workflow file to use this
        from huggingface_hub import snapshot_download
        adapter_cache = {}
        for adapter_id, adapter_name in zip(adapter_ids, adapter_names):
            print(f"copying adapter models {adapter_id} {adapter_name}")
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
    options["option.device_map"] = "auto"

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


def build_rolling_batch_model(model):
    if model not in rolling_batch_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(rolling_batch_model_list.keys())}"
        )
    options = rolling_batch_model_list[model]
    options["option.rolling_batch"] = "scheduler"
    write_model_artifacts(options)


def build_lmi_dist_model(model):
    if model not in lmi_dist_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(lmi_dist_model_list.keys())}"
        )
    options = lmi_dist_model_list[model]
    options["engine"] = "MPI"
    options["option.rolling_batch"] = "lmi-dist"

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

    adapter_ids = options.pop("adapter_ids", [])
    adapter_names = options.pop("adapter_names", [])

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names)


def build_vllm_neo_model(model):
    if model not in vllm_neo_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_neo_model_list.keys())}"
        )
    options = vllm_neo_model_list[model]
    create_neo_input_model(options)


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


def build_trtllm_neo_model(model):
    if model not in trtllm_neo_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(trtllm_neo_list.keys())}"
        )
    options = trtllm_neo_list[model]
    # 60 min timeout for compilation/quantization
    options["model_loading_timeout"] = "3600"
    # Download model to local in addition to generating serving.properties
    create_neo_input_model(options)


def build_transformers_neuronx_neo_model(model):
    if model not in transformers_neuronx_neo_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(transformers_neuronx_neo_list.keys())}"
        )
    options = transformers_neuronx_neo_list[model]
    create_neo_input_model(options)


def build_correctness_model(model):
    if model not in correctness_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(correctness_model_list.keys())}"
        )
    options = correctness_model_list[model]
    write_model_artifacts(options)


def build_handler_performance_model(model):
    if model not in handler_performance_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(handler_performance_model_list.keys())}"
        )
    options = handler_performance_model_list[model]
    write_model_artifacts(options)


def build_text_embedding_model(model):
    if model not in text_embedding_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(onnx_list.keys())}"
        )
    options = text_embedding_model_list[model]
    options["option.task"] = "text_embedding"
    options["normalize"] = False
    write_model_artifacts(options)


supported_handler = {
    'huggingface': build_hf_handler_model,
    'transformers_neuronx': build_transformers_neuronx_handler_model,
    'performance': build_performance_model,
    'handler_performance': build_handler_performance_model,
    'rolling_batch_scheduler': build_rolling_batch_model,
    'lmi_dist': build_lmi_dist_model,
    'lmi_dist_aiccl': build_lmi_dist_aiccl_model,
    'vllm': build_vllm_model,
    'vllm_neo': build_vllm_neo_model,
    'trtllm': build_trtllm_handler_model,
    'trtllm_neo': build_trtllm_neo_model,
    'transformers_neuronx_neo': build_transformers_neuronx_neo_model,
    'correctness': build_correctness_model,
    'text_embedding': build_text_embedding_model,
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
