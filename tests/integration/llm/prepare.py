import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler', help='the handler used in the model')
parser.add_argument('model', help='model that works with certain handler')

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
        "option.device_map": "auto"
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
        "option.task": "text-generation"
    }
}

sd_handler_list = {
    "stable-diffusion-v1-4": {
        "option.s3url": "s3://djl-llm/stable-diffusion-v1-4/",
        "option.tensor_parallel_degree": 2,
        "option.dtype": "fp16"
    },
    "stable-diffusion-v1-5": {
        "option.s3url": "s3://djl-llm/stable-diffusion-v1-5/",
        "option.tensor_parallel_degree": 4,
        "option.dtype": "fp32"
    },
}


def write_prperties(properties):
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
    write_prperties(options)


def build_ds_handler_model(model):
    if model not in ds_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = ds_handler_list[model]
    options["engine"] = "DeepSpeed"
    options["option.entryPoint"] = "djl_python.deepspeed"
    write_prperties(options)


def build_ds_raw_model(model):
    options = ds_model_list[model]
    options["engine"] = "DeepSpeed"
    write_prperties(options)
    shutil.copyfile("llm/deepspeed-model.py", "models/test/model.py")


def build_sd_handler_model(model):
    if model not in sd_handler_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(ds_handler_list.keys())}"
        )
    options = sd_handler_list[model]
    options["engine"] = "DeepSpeed"
    options["option.entryPoint"] = "djl_python.stable-diffusion"
    write_prperties(options)


supported_handler = {
    'deepspeed': build_ds_handler_model,
    'huggingface': build_hf_handler_model,
    "deepspeed_raw": build_ds_raw_model,
    'stable-diffusion': build_sd_handler_model
}

if __name__ == '__main__':
    args = parser.parse_args()
    if args.handler not in supported_handler:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}"
        )
    supported_handler[args.handler](args.model)
