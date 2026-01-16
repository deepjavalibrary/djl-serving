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

performance_test_list = {}

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
    "gpt-neox-20b-custom": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b-custom",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4
    },
    "mistral-7b": {
        "option.model_id": "s3://djl-llm/mistral-7b-instruct-v03",
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
        "option.speculative_config":
        '{"method":"medusa","model":"abhigoyal/vllm-medusa-llama-68m-random","num_speculative_tokens":4}',
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
    },
    "llama3-1-8b-speculative-eagle": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.task": "text-generation",
        "option.speculative_config":
        '{"method":"eagle","model":"yuhuili/EAGLE-LLaMA3.1-Instruct-8B","num_speculative_tokens":4}',
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.enforce_eager": True,
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
    "llama3-8b-unmerged-lora-with-custom-code": {
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
        "add_output_formatter":
        True,
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
        "option.limit_mm_per_prompt": '{"image": 4}',
        "option.gpu_memory_utilization": "0.7",
        "option.enforce_eager": True,
        "option.tensor_parallel_degree": 1,
    },
    "paligemma-3b-mix-448": {
        "option.model_id": "s3://djl-llm/paligemma-3b-mix-448/",
        "option.tensor_parallel_degree": 1,
    },
    "phi-3-vision-128k-instruct": {
        "option.model_id": "s3://djl-llm/phi-3-vision-128k-instruct/",
        "option.limit_mm_per_prompt": '{"image": 4}',
        "option.trust_remote_code": True,
        "option.max_model_len": 8192,
    },
    "pixtral-12b": {
        "option.model_id": "s3://djl-llm/pixtral-12b-2409/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.tokenizer_mode": "mistral",
        "option.limit_mm_per_prompt": '{"image": 4}',
    },
    "llama32-11b-multimodal": {
        "option.model_id": "s3://djl-llm/llama-3-2-11b-vision-instruct/",
        "option.max_model_len": 8192,
        "option.max_rolling_batch_size": 16,
        "option.enforce_eager": True,
    },
    "llama3-1-8b-instruct-tool": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-instruct-hf/",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.enable_auto_tool_choice": True,
        "option.tool_call_parser": "llama3_json",
    },
    "mistral-7b-instruct-v03-tool": {
        "option.model_id": "s3://djl-llm/mistral-7b-instruct-v03/",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.enable_auto_tool_choice": True,
        "option.tool_call_parser": "mistral",
    },
    "deepseek-r1-distill-qwen-1-5b": {
        "option.model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 4,
        "option.enable_reasoning": True,
        "option.reasoning_parser": "deepseek_r1",
    },
    "qwen3-8b": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
    },
    "qwen3-8b-lmcache": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_qwen3_benchmark.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0,
    },
    "qwen3-8b-lmcache-s3": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_s3.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0,
    },
    "qwen3-8b-lmcache-redis": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_redis.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0,
    },
    "qwen3-8b-baseline": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "gpu.maxWorkers": 1,
        "load_on_devices": 0,
    },
    "qwen3-8b-lmcache-ebs": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_qwen3_ebs.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0,
    },
    "qwen3-8b-lmcache-nvme": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_qwen3_nvme.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0,
    },
    "qwen3-8b-no-cache": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.enable_prefix_caching": False,
        "load_on_devices": 0,
    },
    "qwen3-8b-vllm-prefix-cache": {
        "option.model_id": "Qwen/Qwen3-8B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.enable_prefix_caching": True,
        "load_on_devices": 0,
    },
    "qwen2.5-1.5b": {
        "option.model_id": "Qwen/Qwen2.5-1.5B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
    },
    "qwen2.5-7b": {
        "option.model_id": "Qwen/Qwen2.5-7B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
    },
    "qwen2.5-72b": {
        "option.model_id": "Qwen/Qwen2.5-72B",
        "option.tensor_parallel_degree": 8,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
    },
    "qwen2.5-1.5b-lmcache": {
        "option.model_id": "Qwen/Qwen2.5-1.5B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_redis.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0
    },
    "qwen2.5-7b-lmcache": {
        "option.model_id": "Qwen/Qwen2.5-7B",
        "option.tensor_parallel_degree": 1,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_qwen25_7b.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0
    },
    "qwen2.5-72b-lmcache": {
        "option.model_id": "Qwen/Qwen2.5-72B",
        "option.tensor_parallel_degree": 4,
        "option.load_format": "dummy",
        "option.max_new_tokens": 100,
        "option.lmcache_config_file": "lmcache_qwen25_72b.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "load_on_devices": 0
    },
    "tinyllama-input-len-exceeded": {
        "option.model_id": "s3://djl-llm/tinyllama-1.1b-chat/",
        "option.max_model_len": "50",
        "option.max_rolling_batch_size": "1",
        "option.enforce_eager": True,
    },
    "qwen3-vl-32b-instruct": {
        "option.model_id": "s3://djl-llm/Qwen3-VL-32B-Instruct/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 4,
        "option.trust_remote_code": True,
        "option.limit_mm_per_prompt": '{"image": 4, "video": 0}',
    },
    "minimax-m2": {
        "option.model_id": "s3://djl-llm/MiniMax-M2/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 4,
        "option.trust_remote_code": True,
        "option.max_model_len": 16384,
        "option.gpu_memory_utilization": "0.9",
    },
    "llama-4-scout-17b-16e-instruct": {
        "option.model_id": "s3://djl-llm/Llama-4-Scout-17B-16E-Instruct/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 8,
        "option.max_rolling_batch_size": 4,
        "option.trust_remote_code": True,
        "option.max_model_len": 16384,
        "option.gpu_memory_utilization": "0.9",
    },
    "llama3-8b-lmcache-cpu": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        4,
        "option.lmcache_config_file":
        "lmcache_cpu.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    },
    "llama3-8b-lmcache-local-storage": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        4,
        "option.lmcache_config_file":
        "lmcache_local_storage.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    },
    "llama3-8b-lmcache-missing-role": {
        "option.model_id": "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree": 4,
        "option.kv_transfer_config": '{"kv_connector":"LMCacheConnectorV1"}',
    },
    "llama3-8b-no-lmcache": {
        "option.model_id": "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree": 4,
    },
    "llama3-8b-lmcache-s3": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        4,
        "option.lmcache_config_file":
        "lmcache_s3.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    },
    "llama3-8b-lmcache-redis": {
        "option.model_id":
        "s3://djl-llm/llama-3-8b-instruct-hf/",
        "option.tensor_parallel_degree":
        4,
        "option.lmcache_config_file":
        "lmcache_redis.yaml",
        "option.kv_transfer_config":
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    },
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
    "llama-3.1-8b-multi-node-sharding": {
        "option.model_id": "s3://djl-llm/llama-3.1-8b-hf/",
        "option.tensor_parallel_degree": "2",
        "option.pipeline_parallel_degree": "2",
        "option.load_format": "sagemaker_fast_model_loader",
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
        "option.fp8_scheme": "FP8_DYNAMIC"
    }
}

trtllm_handler_list = {
    "llama2-13b": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.tensor_parallel_degree": 4,
    },
    "llama2-7b-smoothquant": {
        "option.model_id": "s3://djl-llm/meta-llama-Llama-2-7b-chat-hf/",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
        "option.smoothquant_per_token": "True",
        "option.smoothquant_per_channel": "True",
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
    "trtllm-llama3-8b": {
        "engine": "Python",
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.rolling_batch": "trtllm",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 213
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
    "tiny-llama-vllm": {
        "engine": "Python",
        "option.rolling_batch": "disable",
        "option.async_mode": True,
        "option.model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "option.gpu_memory_utilization": "0.9",
        "option.max_rolling_batch_size": 512,
        "option.entryPoint": "djl_python.lmi_vllm.vllm_async_service",
    },
    "tiny-llama-trtllm": {
        "engine": "Python",
        "option.model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "option.max_rolling_batch_size": 512,
    },
}

stateful_model_list = {
    "llama3-8b": {
        "option.model_id": "s3://djl-llm/llama-3-8b-hf/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 32,
    },
    "gemma-2b": {
        "option.model_id": "s3://djl-llm/gemma-2b",
        "option.task": "text-generation",
        "option.trust_remote_code": True,
        "option.tensor_parallel_degree": 1,
        "option.max_rolling_batch_size": 32,
    },
}


def create_model_py_with_output_formatter(target_dir, identifier_field,
                                          identifier_value):
    """
    Create a model.py file with a custom output formatter.
    
    Args:
        target_dir: Directory where model.py will be created
        identifier_field: Field name to add to output (e.g., "_model_name", "_adapter_name")
        identifier_value: Value for the identifier field
    """
    # Use triple quotes and avoid f-string for the generated code
    model_py_content = '''"""Custom output formatter"""

from djl_python.output_formatter import output_formatter
import json

@output_formatter
def custom_output_formatter(output, **kwargs):
    """
    Add custom fields
    """
    if hasattr(output, 'model_dump'): # Sync Pydantic Object
        output.{field} = "{value}"
        return output
    elif isinstance(output, str) and output.startswith("data: "): # Streaming SSE String
        if output.strip() != "data: [DONE]":
            data = json.loads(output[6:]) # Parse the JSON data after "data: "
            data["{field}"] = "{value}"
            return f"data: {{json.dumps(data)}}"
    return output
'''.format(field=identifier_field, value=identifier_value)

    model_py_path = os.path.join(target_dir, "model.py")
    with open(model_py_path, "w") as f:
        f.write(model_py_content)


def write_model_artifacts(properties,
                          requirements=None,
                          adapter_ids=[],
                          adapter_names=[],
                          lmcache_config_file=None,
                          add_output_formatter=False):
    model_path = "models/test"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    if lmcache_config_file:
        source_config = os.path.join("lmcache_configs", lmcache_config_file)
        dest_config = os.path.join(model_path, lmcache_config_file)
        if os.path.exists(source_config):
            shutil.copy2(source_config, dest_config)

    with open(os.path.join(model_path, "serving.properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key}={value}\n")
    if requirements:
        with open(os.path.join(model_path, "requirements.txt"), "w") as f:
            f.write('\n'.join(requirements) + '\n')

    # Add base model output formatter if requested
    if add_output_formatter:
        model_id = properties.get("option.model_id", "unknown_model")
        create_model_py_with_output_formatter(model_path, "processed_by",
                                              model_id)

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

                # Add adapter-specific output formatter if requested
                if add_output_formatter:
                    create_model_py_with_output_formatter(
                        dir, "processed_by", adapter_name)


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


def build_vllm_async_model(model):
    if model not in vllm_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_model_list.keys())}"
        )
    options = vllm_model_list[model]
    options["engine"] = "Python"

    adapter_ids = options.pop("adapter_ids", [])
    adapter_names = options.pop("adapter_names", [])
    lmcache_config_file = options.get("option.lmcache_config_file", None)

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names,
                          lmcache_config_file=lmcache_config_file,
                          add_output_formatter=options.pop(
                              "add_output_formatter", False))


def build_vllm_async_model_with_custom_handler(model, handler_type="success"):
    if model not in vllm_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_model_list.keys())}"
        )
    options = vllm_model_list[model]
    options["engine"] = "Python"
    options["option.rolling_batch"] = "disable"
    options["option.async_mode"] = "true"
    options["option.entryPoint"] = "djl_python.lmi_vllm.vllm_async_service"
    write_model_artifacts(options)

    # Copy custom handler from examples
    source_file = f"examples/custom_handlers/{handler_type}.py"
    target_file = "models/test/model.py"
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)


def build_vllm_async_model_custom_formatters(model, error_type=None):
    if model not in vllm_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_model_list.keys())}"
        )
    options = vllm_model_list[model]
    options["engine"] = "Python"
    options["option.rolling_batch"] = "disable"
    options["option.async_mode"] = "true"
    options["option.entryPoint"] = "djl_python.lmi_vllm.vllm_async_service"
    write_model_artifacts(options)

    # Create custom formatter files based on error_type
    source_dir = "examples/custom_formatters/"
    target_dir = "models/test/"

    if not error_type:
        source_dir = "examples/custom_formatters/"
        target_dir = "models/test/"
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, filename)
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)
        return
    elif error_type == "input":
        filename = "input_formatter_failed.py"
    elif error_type == "output":
        # Create model.py with failing output formatter
        filename = "output_formatter_failed.py"
    elif error_type == "load":
        # Create model.py with syntax error to cause load failure
        filename = "load_formatter_failed.py"
    if os.path.exists(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, "model.py")
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)


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

    if len(adapter_ids) == 0:
        # use async mode for non lora tests
        build_vllm_async_model(model)
        return

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


def build_trtllm_handler_model(model):
    if model not in trtllm_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(trtllm_handler_list.keys())}"
        )
    options = trtllm_handler_list[model]
    options["option.rolling_batch"] = "disable"
    options["option.async_mode"] = True
    options["option.entryPoint"] = "djl_python.lmi_trtllm.trtllm_async_service"
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
            f"{model} is not one of the supporting handler {list(text_embedding_model_list.keys())}"
        )
    options = text_embedding_model_list[model]
    options["option.task"] = "text_embedding"
    options["normalize"] = False
    write_model_artifacts(options)


def build_stateful_model(model):
    if model not in stateful_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(text_embedding_model_list.keys())}"
        )
    options = stateful_model_list[model]
    options["engine"] = "Python"
    options["option.rolling_batch"] = "disable"
    options["option.async_mode"] = "true"
    options["option.entryPoint"] = "djl_python.lmi_vllm.vllm_async_service"
    options["option.enable_stateful_sessions"] = "true"
    options["option.sessions_path"] = "/tmp/djl_sessions"
    write_model_artifacts(options)


supported_handler = {
    'huggingface': build_hf_handler_model,
    'performance': build_performance_model,
    'handler_performance': build_handler_performance_model,
    'vllm': build_vllm_model,
    'vllm_neo': build_vllm_neo_model,
    'trtllm': build_trtllm_handler_model,
    'trtllm_neo': build_trtllm_neo_model,
    'correctness': build_correctness_model,
    'text_embedding': build_text_embedding_model,
    'vllm_async': build_vllm_async_model,
    'vllm_async_custom_formatters': build_vllm_async_model_custom_formatters,
    'vllm_async_custom_handler': build_vllm_async_model_with_custom_handler
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
