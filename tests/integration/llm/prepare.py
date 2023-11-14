import argparse
import os
import shutil

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
args = parser.parse_args()

ds_aot_list = {
    "gpt-neo-2.7b": {
        "option.model_id":
        "EleutherAI/gpt-neo-2.7B",
        "option.tensor_parallel_degree":
        2,
        "option.task":
        "text-generation",
        "option.dtype":
        "float16",
        "option.save_mp_checkpoint_path":
        "/opt/ml/input/data/training/partition-test"
    },
}

ds_aot_handler_list = {
    "opt-6.7b": {
        "option.model_id":
        "s3://djl-llm/opt-6b7/",
        "option.tensor_parallel_degree":
        4,
        "option.task":
        "text-generation",
        "option.dtype":
        "fp16",
        "option.save_mp_checkpoint_path":
        "/opt/ml/input/data/training/partition-test"
    },
    "bloom-7b1": {
        "option.model_id":
        "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree":
        4,
        "option.task":
        "text-generation",
        "option.dtype":
        "fp16",
        "option.save_mp_checkpoint_path":
        "s3://djl-llm/bloom-7b1-tp4/ds-aot-handler/"
    }
}

ds_model_list = {
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.tensor_parallel_degree": 4
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "float16"
    },
    "opt-30b": {
        "option.model_id": "s3://djl-llm/opt-30b/",
        "option.tensor_parallel_degree": 4
    }
}

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
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b/",
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
        "option.model_id": "nomic-ai/gpt4all-lora",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "fp16"
    },
    "llama-7b-unmerged-lora": {
        "option.model_id": "s3://djl-llm/huggyllama-llama-7b",
        "option.tensor_parallel_degree": 1,
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.adapters": "adapters",
        "adapter_ids": ["tloen/alpaca-lora-7b", "22h/cabrita-lora-v0-1"],
        "adapter_names": ["english-alpaca", "portugese-alpaca"],
    }
}

ds_handler_list = {
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.enable_streaming": False
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.enable_streaming": False
    },
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16",
        "option.enable_streaming": False
    },
    "opt-13b": {
        "option.model_id": "s3://djl-llm/opt-13b/",
        "option.tensor_parallel_degree": 2,
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.enable_streaming": False
    },
    "gpt-neo-1.3b": {
        "option.model_id": "EleutherAI/gpt-neo-1.3B",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "option.enable_streaming": True
    },
    "gpt4all-lora": {
        "option.model_id": "nomic-ai/gpt4all-lora",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.enable_streaming": False
    }
}

sd_handler_list = {
    "stable-diffusion-v1-5": {
        "option.model_id": "s3://djl-llm/stable-diffusion-v1-5/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16"
    },
    "stable-diffusion-2-1-base": {
        "option.model_id": "s3://djl-llm/stable-diffusion-2-1-base/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16"
    },
    "stable-diffusion-2-depth": {
        "option.model_id": "s3://djl-llm/stable-diffusion-2-depth/",
        "option.tensor_parallel_degree": 1,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1
    }
}

default_accel_configs = {
    "huggingface": {
        "engine": "Python",
        "option.entryPoint": "djl_python.huggingface"
    },
    "deepspeed": {
        "engine": "DeepSpeed",
        "option.entryPoint": "djl_python.deepspeed"
    }
}

performance_test_list = {
    "opt-30b": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/opt-30b/"
    },
    "open-llama-13b-fp16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/open-llama-13b/"
    },
    "open-llama-13b-fp16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/open-llama-13b/"
    },
    "open-llama-13b-bf16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/open-llama-13b/"
    },
    "open-llama-13b-bf16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/open-llama-13b/"
    },
    "open-llama-13b-smoothquant": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/open-llama-13b/",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.quantize": "smoothquant"
    },
    "gpt-j-6b-fp16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-j-6b/"
    },
    "gpt-j-6b-fp16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/gpt-j-6b/"
    },
    "gpt-j-6b-bf16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-j-6b/"
    },
    "gpt-j-6b-bf16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/gpt-j-6b/"
    },
    "gpt-j-6b-smoothquant": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.quantize": "smoothquant"
    },
    "bloom-7b1": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/bloom-7b1/"
    },
    "gpt-neox-20b-fp16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/"
    },
    "gpt-neox-20b-fp16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/"
    },
    "gpt-neox-20b-bf16-deepspeed": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/"
    },
    "gpt-neox-20b-bf16-huggingface": {
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.engine": "huggingface",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/"
    },
    "gpt-neox-20b-smoothquant": {
        "option.task": "text-generation",
        "option.dtype": "fp16",
        "option.engine": "deepspeed",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/",
        "option.quantize": "smoothquant",
        "option.smoothquant_alpha": 0.65
    }
}

transformers_neuronx_handler_list = {
    "gpt2": {
        "option.model_id": "gpt2",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": False
    },
    "gpt2-quantize": {
        "option.model_id": "gpt2",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.load_in_8bit": True,
        "option.enable_streaming": False
    },
    "opt-1.3b": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": False
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.dtype": "fp32",
        "option.model_loading_timeout": 900,
        "option.enable_streaming": False
    },
    "pythia-2.8b": {
        "option.model_id": "s3://djl-llm/pythia-2.8b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 900,
        "option.enable_streaming": False
    },
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.neuron_optimize_level": 1,
        "option.model_loading_timeout": 1200,
        "option.enable_streaming": False
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 256,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 720,
        "option.enable_streaming": False
    },
    "llama-7b-split": {
        "option.model_id": "s3://djl-llm/llama-2-7b-split-inf2/split-model/",
        "option.batch_size": 1,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.model_loading_timeout": 2400,
        "option.load_split_model": True,
    },
    "opt-1.3b-streaming": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "option.batch_size": 2,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": True,
    },
    "stable-diffusion-2.1-base-neuron": {
        "option.model_id": "s3://djl-llm/stable-diffusion-2-1-base-compiled/",
        "option.tensor_parallel_degree": 2,
        "option.use_stable_diffusion": True
    },
    "stable-diffusion-2.1-base-neuron-bf16": {
        "option.model_id":
        "s3://djl-llm/stable-diffusion-2-1-base-compiled-bf16/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16",
        "option.use_stable_diffusion": True
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
        "option.model_id": "TheBloke/Llama-2-7b-Chat-GPTQ",
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
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
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
    "gpt-neox-20b-bitsandbytes": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "bitsandbytes"
    },
    "llama2-13b-gptq": {
        "option.model_id": "TheBloke/Llama-2-13B-chat-GPTQ",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "gptq"
    },
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
}

unmerged_lora_correctness_list = {
    "llama-7b-unmerged-lora": {
        "option.tensor_parallel_degree": 1,
        "gpu.maxWorkers": 1,
        "load_on_devices": 0,
    }
}

ds_smoothquant_model_list = {
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
    },
    "gpt-neox-20b": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
        "option.smoothquant_alpha": 0.65,
    },
    "llama2-13b-dynamic-int8": {
        "option.model_id": "TheBloke/Llama-2-13B-fp16",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "dynamic_int8",
    },
    "llama2-13b-smoothquant": {
        "option.model_id": "TheBloke/Llama-2-13B-fp16",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
    },
}

lmi_dist_aiccl_model_list = {
    "llama-2-70b-aiccl": {
        "option.model_id": "s3://djl-llm/llama-2-70b-hf/",
    },
    "codellama-34b-aiccl": {
        "option.model_id": "codellama/CodeLlama-34b-hf",
    },
    "falcon-40b-aiccl": {
        "option.model_id": "tiiuae/falcon-40b",
    },
}

trtllm_handler_list = {
    "llama2-13b": {
        "option.model_id": "s3://djl-llm/llama-2-13b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.output_formatter": "jsonlines",
    },
    "falcon-7b": {
        "option.model_id": "s3://djl-llm/triton/falcon-7b-tp1-bs4/",
        "option.tensor_parallel_degree": 1,
        "option.output_formatter": "jsonlines",
    },
    "llama2-7b-smoothquant": {
        "option.model_id": "s3://djl-llm/llama-2-7b-hf/",
        "option.tensor_parallel_degree": 4,
        "option.quantize": "smoothquant",
        "option.smoothquant_per_token": "True",
        "option.smoothquant_per_channel": "True",
        "option.output_formatter": "jsonlines",
    },
}

deepspeed_rolling_batch_model_list = {
    "gpt-neox-20b": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4
    },
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b",
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
    "llama2-13b-smoothquant": {
        "option.model_id": "TheBloke/Llama-2-13B-fp16",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "smoothquant",
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
        for adapter_id, adapter_name in zip(adapter_ids, adapter_names):
            os.makedirs(os.path.join(adapters_path, adapter_name),
                        exist_ok=True)
            snapshot_download(adapter_id,
                              local_dir=os.path.join(adapters_path,
                                                     adapter_name))


def build_hf_handler_model(model):
    if model not in hf_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(hf_handler_list.keys())}"
        )
    options = hf_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.huggingface"
    options["option.predict_timeout"] = 240

    adapter_ids = []
    adapter_names = []
    if "option.adapters" in options:
        adapter_ids = options["adapter_ids"]
        adapter_names = options["adapter_names"]
        del options["adapter_ids"]
        del options["adapter_names"]

    write_model_artifacts(options,
                          adapter_ids=adapter_ids,
                          adapter_names=adapter_names)


def build_ds_handler_model(model):
    if model not in ds_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = ds_handler_list[model]
    options["engine"] = "DeepSpeed"
    # options["option.entryPoint"] = "djl_python.deepspeed"
    write_model_artifacts(options)


def build_ds_raw_model(model):
    options = ds_model_list[model]
    options["engine"] = "DeepSpeed"
    write_model_artifacts(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


def build_ds_aot_model(model):
    if model not in ds_aot_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_aot_list.keys())}"
        )

    options = ds_aot_list[model]
    options["engine"] = "DeepSpeed"
    write_model_artifacts(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


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
    engine = options.get('option.engine')
    if args.engine:
        engine = args.engine
    if engine is None:
        raise ValueError("Need to provide engine for performance benchmark")
    for k, v in default_accel_configs[engine].items():
        if k not in options:
            options[k] = v
    write_model_artifacts(options)


def build_ds_aot_handler_model(model):
    if model not in ds_aot_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_aot_handler_list.keys())}"
        )

    options = ds_aot_handler_list[model]
    options["engine"] = "DeepSpeed"
    write_model_artifacts(options)


def build_sd_handler_model(model):
    if model not in sd_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = sd_handler_list[model]
    options["engine"] = "DeepSpeed"
    options["option.entryPoint"] = "djl_python.stable-diffusion"
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
    write_model_artifacts(options)


def build_vllm_model(model):
    if model not in vllm_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(vllm_model_list.keys())}"
        )
    options = vllm_model_list[model]
    options["engine"] = "Python"
    options["option.rolling_batch"] = "vllm"
    options["option.output_formatter"] = "jsonlines"
    write_model_artifacts(options)


def build_unmerged_lora_correctness_model(model):
    if model not in unmerged_lora_correctness_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(unmerged_lora_correctness_list.keys())}"
        )
    options = unmerged_lora_correctness_list[model]
    options["engine"] = "Python"
    write_model_artifacts(options)
    shutil.copyfile("llm/unmerged_lora.py", "models/test/model.py")


def build_ds_smoothquant_model(model):
    if model not in ds_smoothquant_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_smoothquant_model_list.keys())}"
        )
    options = ds_smoothquant_model_list[model]
    options["engine"] = "DeepSpeed"
    options["entryPoint"] = "djl_python.deepspeed"
    options["dtype"] = "fp16"
    options["task"] = "text-generation"
    write_model_artifacts(options)


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
    options["option.max_rolling_batch_size"] = 4
    write_model_artifacts(options)


def build_trtllm_handler_model(model):
    if model not in trtllm_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(trtllm_handler_list.keys())}"
        )
    options = trtllm_handler_list[model]
    write_model_artifacts(options)


def build_deepspeed_rolling_batch_model(model):
    if model not in deepspeed_rolling_batch_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(deepspeed_rolling_batch_model_list.keys())}"
        )
    options = deepspeed_rolling_batch_model_list[model]
    options["engine"] = "DeepSpeed"
    options["option.rolling_batch"] = "deepspeed"
    options["option.output_formatter"] = "jsonlines"
    write_model_artifacts(options)


supported_handler = {
    'deepspeed': build_ds_handler_model,
    'huggingface': build_hf_handler_model,
    "deepspeed_raw": build_ds_raw_model,
    'stable-diffusion': build_sd_handler_model,
    'deepspeed_aot': build_ds_aot_model,
    'deepspeed_handler_aot': build_ds_aot_handler_model,
    'transformers_neuronx': build_transformers_neuronx_handler_model,
    'performance': build_performance_model,
    'rolling_batch_scheduler': build_rolling_batch_model,
    'lmi_dist': build_lmi_dist_model,
    'vllm': build_vllm_model,
    'unmerged_lora': build_unmerged_lora_correctness_model,
    'deepspeed_smoothquant': build_ds_smoothquant_model,
    'lmi_dist_aiccl': build_lmi_dist_aiccl_model,
    'trtllm': build_trtllm_handler_model,
    'deepspeed_rolling_batch': build_deepspeed_rolling_batch_model,
}

if __name__ == '__main__':
    if args.handler not in supported_handler:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}"
        )
    supported_handler[args.handler](args.model)
