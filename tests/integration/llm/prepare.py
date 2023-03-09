import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler', help='the handler used in the model')
parser.add_argument('model', help='model that works with certain handler')

ds_aot_list = {
    "opt-6.7b": {
        "option.s3url": "s3://djl-llm/opt-6b7/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "float16"
    },
    "gpt-neox-20b": {
        "option.s3url": "s3://djl-llm/gpt-neox-20b/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "float16"
    }
}

ds_model_list = {
    "gpt-j-6b": {
        "option.s3url": "s3://djl-llm/gpt-j-6b/",
        "option.tensor_parallel_degree": 4
    },
    "bloom-7b1": {
        "option.s3url": "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "float16"
    },
    "opt-30b": {
        "option.s3url": "s3://djl-llm/opt-30b/",
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
        "option.s3url": "s3://djl-llm/gpt-j-6b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.device_map": "auto",
        "option.dtype": "fp16"
    },
    "bloom-7b1": {
        "option.s3url": "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.load_in_8bit": "TRUE",
        "option.device_map": "auto"
    }
}

ds_handler_list = {
    "gpt-j-6b": {
        "option.s3url": "s3://djl-llm/gpt-j-6b/",
        "option.task": "text-generation",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "bf16"
    },
    "bloom-7b1": {
        "option.s3url": "s3://djl-llm/bloom-7b1/",
        "option.tensor_parallel_degree": 4,
        "option.task": "text-generation",
        "option.dtype": "fp16"
    },
    "opt-13b": {
        "option.s3url": "s3://djl-llm/opt-13b/",
        "option.tensor_parallel_degree": 2,
        "option.task": "text-generation",
        "option.dtype": "fp16"
    }
}

sd_handler_list = {
    "stable-diffusion-v1-5": {
        "option.s3url": "s3://djl-llm/stable-diffusion-v1-5/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16"
    },
    "stable-diffusion-2-1-base": {
        "option.s3url": "s3://djl-llm/stable-diffusion-2-1-base/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16"
    },
    "stable-diffusion-2-depth": {
        "option.s3url": "s3://djl-llm/stable-diffusion-2-depth/",
        "option.tensor_parallel_degree": 1,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1
    }
}

ft_model_list = {
    "t5-small": {
        "option.model_id": "t5-small",
        "option.tensor_parallel_degree": 4,
    },
    "gpt2-xl": {
        "option.model_id": "gpt2-xl",
        "option.tensor_parallel_degree": 1,
    },
    "facebook/opt-6.7b": {
        "option.s3url": "s3://djl-llm/opt-6b7/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp16",
    },
    "bigscience/bloom-3b": {
        "option.s3url": "s3://djl-llm/bloom-3b/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16",
        "gpu.maxWorkers": 1,
    },
    "flan-t5-xxl": {
        "option.s3url": "s3://djl-llm/flan-t5-xxl/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp32"
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
    "opt-1.3b": {
        "option.s3url": "s3://djl-llm/opt-1.3b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 4,
        "option.n_positions": 256,
        "option.dtype": "fp16",
        "option.model_loading_timeout": 600
    },
    "gpt-j-6b": {
        "option.s3url": "s3://djl-llm/gpt-j-6b/",
        "option.batch_size": 4,
        "option.tensor_parallel_degree": 8,
        "option.n_positions": 512,
        "option.dtype": "fp32",
        "option.model_loading_timeout": 720
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
    options["option.entryPoint"] = "djl_python.deepspeed"
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
    options["option.save_mp_checkpoint_path"] = "/opt/ml/model/partition-test"
    write_properties(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


def build_sd_handler_model(model):
    if model not in sd_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = sd_handler_list[model]
    options["engine"] = "DeepSpeed"
    options["option.entryPoint"] = "djl_python.stable-diffusion"
    write_properties(options)


def build_ft_raw_model(model):
    if model not in ft_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_model_list.keys())}"
        )
    options = ft_model_list[model]
    options["engine"] = "FasterTransformer"
    write_properties(options)
    shutil.copyfile("llm/fastertransformer-model.py", "models/test/model.py")


def build_ft_raw_aot_model(model):
    if model not in ft_model_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ft_model_list.keys())}"
        )
    options = ft_model_list[model]
    options["engine"] = "FasterTransformer"
    options["option.save_mp_checkpoint_path"] = "/opt/ml/model/partition-test"
    write_properties(options)
    shutil.copyfile("llm/fastertransformer-model.py", "models/test/model.py")


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


supported_handler = {
    'deepspeed': build_ds_handler_model,
    'huggingface': build_hf_handler_model,
    "deepspeed_raw": build_ds_raw_model,
    'stable-diffusion': build_sd_handler_model,
    'fastertransformer_raw': build_ft_raw_model,
    'fastertransformer_raw_aot': build_ft_raw_aot_model,
    'deepspeed_aot': build_ds_aot_model,
    'transformers_neuronx_raw': build_transformers_neuronx_model,
    'transformers_neuronx': build_transformers_neuronx_handler_model
}

if __name__ == '__main__':
    args = parser.parse_args()
    if args.handler not in supported_handler:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}"
        )
    supported_handler[args.handler](args.model)
