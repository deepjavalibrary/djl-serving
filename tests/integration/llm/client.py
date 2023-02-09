import requests
import argparse
import subprocess as sp
import logging
import math

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler', help='the handler used in the model')
parser.add_argument('model', help='The name of model')

endpoint = "http://127.0.0.1:8080/predictions/test"

ds_raw_model_spec = {
    "gpt-j-6b": {
        "max_memory_per_gpu": 10.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": True
    },
    "bloom-7b1": {
        "max_memory_per_gpu": 10.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": False
    },
    "opt-30b": {
        "max_memory_per_gpu": 16.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": False
    }
}

hf_model_spec = {
    "gpt-neo-2.7b": {
        "max_memory_per_gpu": 8.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "gpt-j-6b": {
        "max_memory_per_gpu": 14.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "bloom-7b1": {
        "max_memory_per_gpu": 10.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128]
    }
}

ds_model_spec = {
    "gpt-j-6b": {
        "max_memory_per_gpu": 10.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "bloom-7b1": {
        "max_memory_per_gpu": 10.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256]
    },
    "opt-13b": {
        "max_memory_per_gpu": 15.0,
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    }
}

ft_raw_model_spec = {
    "t5-small": {
        "batch_size": [1, 2]
    }
}

sd_model_spec = {
    "stable-diffusion-v1-4": {
        "max_memory_per_gpu": 8.0,
        "size": [256, 512],
        "steps": [1, 2],
        "worker": 2
    },
    "stable-diffusion-v1-5": {
        "max_memory_per_gpu": 16.0,
        "size": [256, 512],
        "steps": [1, 2]
    },
}


def check_worker_number(desired):
    endpoint = "http://127.0.0.1:8080/models/test"
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
    res = requests.post(endpoint, headers=headers, json=data)
    return res


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]


def batch_generation(batch_size):
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

def t5_batch_generation(batch_size):
    input_sentences = [
        "translate English to German: The house is wonderful.",
        "translate English to German: My name is AWS",
    ]
    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    return input_sentences[:batch_size]

def test_handler(model, model_spec):
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
            req["parameters"] = params
            logging.info(f"req {req}")
            res = send_json(req)
            res = res.json()
            logging.info(f"res {res}")
            result = [item[0]['generated_text'] for item in res]
            assert len(result) == batch_size
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


def test_ds_raw_model(model):
    if model not in ds_raw_model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(ds_raw_model_spec.keys())}"
        )
    spec = ds_raw_model_spec[args.model]
    for batch_size in spec["batch_size"]:
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
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


def test_sd_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(sd_model_spec.keys())}"
        )
    spec = sd_model_spec[model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    from PIL import Image
    from io import BytesIO
    for size in spec["size"]:
        for step in spec["steps"]:
            req = {"prompt": "A bird and cat flying through space"}
            params = {"height": size, "width": size, "steps": step}
            req["parameters"] = params
            logging.info(f"req: {req}")
            res = send_json(req)
            assert res.status_code == 200
            try:
                img = Image.open(BytesIO(res.content)).convert("RGB")
            except Exception as e:
                raise IOError("failed to deserialize image from response", e)
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]

def test_ft_raw_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(ft_raw_model_spec.keys())}"
        )
    spec = model_spec[model]
    for batch_size in spec['batch_size']:
        print(f"testing ft_handler with model: {model}, batch_size: {batch_size} ")
        req = {"inputs" : t5_batch_generation(batch_size)}
        res = send_json(req)
        res = res.json()
        assert len(res) == batch_size

if __name__ == '__main__':
    args = parser.parse_args()
    if args.handler == "deepspeed_raw":
        test_ds_raw_model(args.model)
    elif args.handler == "huggingface":
        test_handler(args.model, hf_model_spec)
    elif args.handler == "deepspeed":
        test_handler(args.model, ds_model_spec)
    elif args.handler == "stable-diffusion":
        test_sd_handler(args.model, sd_model_spec)
    elif args.handler == "fastertransformer_raw":
        test_ft_raw_handler(args.model, ft_raw_model_spec)
    else:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler")
