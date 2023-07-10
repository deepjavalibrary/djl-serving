import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler', help='the handler used in the model')
parser.add_argument('model', help='model that works with certain handler')
parser.add_argument('--engine',
                    required=False,
                    type=str,
                    choices=['deepspeed', 'huggingface', 'fastertransformer'],
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
    }
}

ds_handler_list = {
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16"
    },
    "bloom-7b1": {
        "option.model_id": "s3://djl-llm/bloom-7b1/",
        "option.task": "text-generation",
        "option.dtype": "fp16"
    },
    "open-llama-7b": {
        "option.model_id": "s3://djl-llm/open-llama-7b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16"
    },
    "opt-13b": {
        "option.model_id": "s3://djl-llm/opt-13b/",
        "option.tensor_parallel_degree": 2,
        "option.task": "text-generation",
        "option.dtype": "fp16"
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
        "option.dtype": "fp16"
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

ft_handler_list = {
    "bigscience/bloom-3b": {
        "option.model_id": "s3://djl-llm/bloom-3b/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1,
    },
    "flan-t5-xxl": {
        "option.model_id": "s3://djl-llm/flan-t5-xxl/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp32"
    },
    "EleutherAI/pythia-2.8b": {
        "option.model_id": "s3://djl-llm/pythia-2.8b/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1
    }
}

ft_model_list = {
    "t5-small": {
        "engine": "FasterTransformer",
        "option.model_id": "t5-small",
    },
    "gpt2-xl": {
        "engine": "FasterTransformer",
        "option.model_id": "gpt2-xl",
        "option.tensor_parallel_degree": 1,
    },
    "facebook/opt-6.7b": {
        "engine": "FasterTransformer",
        "option.model_id": "s3://djl-llm/opt-6b7/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16",
    },
    "bigscience/bloom-3b": {
        "engine": "FasterTransformer",
        "option.model_id": "s3://djl-llm/bloom-3b/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1,
    },
    "nomic-ai/gpt4all-j": {
        "option.model_id": "s3://djl-llm/gpt4all-j/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp32"
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
    },
    "fastertransformer": {
        "engine": "FasterTransformer",
        "entryPoint": "djl_python.fastertransformer"
    }
}

performance_test_list = {
    "opt-30b": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/opt-30b/"
    },
    "open-llama-13b": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/open-llama-13b/"
    },
    "gpt-j-6b": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/gpt-j-6b/"
    },
    "bloom-7b1": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/bloom-7b1/"
    },
    "gpt-neox-20b": {
        "option.task": "text-generation",
        "option.model_id": "s3://djl-llm/gpt-neox-20b/"
    }
}

transformers_neuronx_model_list = {
    "gpt2": {
        "option.tensor_parallel_degree": 2,
        "option.batch_size": 4,
        "option.dtype": "f32",
        "option.n_positions": 128,
        "load_on_devices": "nc0"
    },
}

transformers_neuronx_handler_list = {
    "gpt2": {
        "option.model_id": "gpt2",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 2,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "opt-1.3b": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt-j-6b": {
        "option.model_id": "s3://djl-llm/gpt-j-6b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 1024,
        "option.dtype": "fp32",
        "option.model_loading_timeout": 720
    },
    "opt-1.3b-streaming": {
        "option.model_id": "s3://djl-llm/opt-1.3b/",
        "option.batch_size": 2,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 512,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600,
        "option.enable_streaming": True
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
        "option.max_rolling_batch_size": 4
    },
    "gpt-neox-20b-bits-and-bytes-quantized": {
        "option.model_id": "s3://djl-llm/gpt-neox-20b",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 4,
        "option.max_rolling_batch_size": 4,
        "option.quantize": "bitsandbytes",
        "option.trust_remote_code": True
    },
}


def write_properties(properties):
    model_path = "models/test"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "serving.properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key}={value}\n")


def build_hf_handler_model(model):
    if model not in hf_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(hf_handler_list.keys())}"
        )
    options = hf_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.huggingface"
    options["option.predict_timeout"] = 240
    write_properties(options)


def build_ds_handler_model(model):
    if model not in ds_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = ds_handler_list[model]
    options["engine"] = "DeepSpeed"
    # options["option.entryPoint"] = "djl_python.deepspeed"
    write_properties(options)


def build_ds_raw_model(model):
    options = ds_model_list[model]
    options["engine"] = "DeepSpeed"
    write_properties(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


def build_ds_aot_model(model):
    if model not in ds_aot_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_aot_list.keys())}"
        )

    options = ds_aot_list[model]
    options["engine"] = "DeepSpeed"
    write_properties(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


def build_performance_model(model):
    if model in performance_test_list.keys():
        options = performance_test_list[model]
    else:
        options = {"option.task": "text-generation", "option.model_id": model}
    options["option.predict_timeout"] = 240
    options["option.dtype"] = args.dtype
    options["option.tensor_parallel_degree"] = args.tensor_parallel
    for k, v in default_accel_configs[args.engine].items():
        if k not in options:
            options[k] = v
    write_properties(options)


def build_ds_aot_handler_model(model):
    if model not in ds_aot_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_aot_handler_list.keys())}"
        )

    options = ds_aot_handler_list[model]
    options["engine"] = "DeepSpeed"
    write_properties(options)


def build_sd_handler_model(model):
    if model not in sd_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = sd_handler_list[model]
    options["engine"] = "DeepSpeed"
    options["option.entryPoint"] = "djl_python.stable-diffusion"
    write_properties(options)


def build_ft_handler_model(model):
    if model not in ft_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_handler_list.keys())}"
        )
    options = ft_handler_list[model]
    options["engine"] = "FasterTransformer"
    # options["option.entryPoint"] = "djl_python.fastertransformer"
    write_properties(options)


def build_ft_raw_model(model):
    if model not in ft_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_model_list.keys())}"
        )
    options = ft_model_list[model]
    if "engine" not in options:
        options["engine"] = "Python"

    write_properties(options)
    shutil.copyfile("llm/fastertransformer-model.py", "models/test/model.py")


def build_ft_raw_aot_model(model):
    if model not in ft_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_model_list.keys())}"
        )
    options = ft_model_list[model]
    options["engine"] = "FasterTransformer"
    if model == 't5-small':
        options[
            "option.save_mp_checkpoint_path"] = "s3://djl-llm/t5-small-tp4/ft-aot/"
    else:
        options[
            "option.save_mp_checkpoint_path"] = "/opt/ml/input/data/training/partition-test"
    write_properties(options)
    shutil.copyfile("llm/fastertransformer-model.py", "models/test/model.py")


def builder_ft_handler_aot_model(model):
    if model not in ft_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_model_list.keys())}"
        )
    options = ft_model_list[model]
    options["engine"] = "FasterTransformer"
    # options["entryPoint"] = "djl_python.fastertransformer"
    if model == 't5-small':
        options[
            "option.save_mp_checkpoint_path"] = "s3://djl-llm/t5-small-tp4/ft-aot-handler/"
    else:
        options[
            "option.save_mp_checkpoint_path"] = "/opt/ml/input/data/training/partition-test"
    write_properties(options)


def build_transformers_neuronx_model(model):
    if model not in transformers_neuronx_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(transformers_neuronx_model_list.keys())}"
        )
    options = transformers_neuronx_model_list[model]
    options["engine"] = "Python"
    write_properties(options)
    shutil.copyfile("llm/transformers-neuronx-gpt2-model.py",
                    "models/test/model.py")


def build_transformers_neuronx_handler_model(model):
    if model not in transformers_neuronx_handler_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(transformers_neuronx_handler_list.keys())}"
        )
    options = transformers_neuronx_handler_list[model]
    options["engine"] = "Python"
    options["option.entryPoint"] = "djl_python.transformers-neuronx"
    write_properties(options)


def build_rolling_batch_model(model):
    if model not in rolling_batch_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(rolling_batch_model_list.keys())}"
        )
    options = rolling_batch_model_list[model]
    options["rolling_batch"] = "scheduler"
    write_properties(options)


def build_lmi_dist_model(model):
    if model not in lmi_dist_model_list.keys():
        raise ValueError(
            f"{model} is not one of the supporting handler {list(lmi_dist_model_list.keys())}"
        )
    options = lmi_dist_model_list[model]
    options["engine"] = "MPI"
    options["option.rolling_batch"] = "lmi-dist"
    write_properties(options)


supported_handler = {
    'deepspeed': build_ds_handler_model,
    'huggingface': build_hf_handler_model,
    "deepspeed_raw": build_ds_raw_model,
    'stable-diffusion': build_sd_handler_model,
    'fastertransformer': build_ft_handler_model,
    'fastertransformer_raw': build_ft_raw_model,
    'fastertransformer_raw_aot': build_ft_raw_aot_model,
    'fastertransformer_handler_aot': builder_ft_handler_aot_model,
    'deepspeed_aot': build_ds_aot_model,
    'deepspeed_handler_aot': build_ds_aot_handler_model,
    'transformers_neuronx_raw': build_transformers_neuronx_model,
    'transformers_neuronx': build_transformers_neuronx_handler_model,
    'performance': build_performance_model,
    'rolling_batch_scheduler': build_rolling_batch_model,
    'lmi_dist': build_lmi_dist_model
}

if __name__ == '__main__':
    if args.handler not in supported_handler:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}"
        )
    supported_handler[args.handler](args.model)
