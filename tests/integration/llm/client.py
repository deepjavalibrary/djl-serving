import requests
import argparse
import subprocess as sp
import logging
import math

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler',
                    help='the handler used in the model')
parser.add_argument('model',
                    help='The name of model')

endpoint = "http://127.0.0.1:8080/predictions/test"

ds_raw_model_spec = {
    "gpt-j-6b": {"max_memory_per_gpu": 10.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256],
                 "use_pipeline": True},
    "bloom-7b1": {"max_memory_per_gpu": 10.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256],
                  "use_pipeline": False},
    "opt-30b": {"max_memory_per_gpu": 16.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256],
                "use_pipeline": False}
}

hf_model_spec = {
    "gpt-neo-2.7b": {"max_memory_per_gpu": 10.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256]},
    "gpt-j-6b": {"max_memory_per_gpu": 10.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256], "worker": 2},
    "bloom-7b1": {"max_memory_per_gpu": 10.0, "batch_size": [1, 2, 4, 8], "seq_length": [64, 128, 256]}
}


def check_worker_number(desired):
    endpoint = "http://127.0.0.1:8080/models/test"
    res = requests.get(endpoint).json()
    assert desired == len(res[0]["models"][0]["workerGroups"][0]["workers"])


def send_json(data):
    headers = {'content-type': 'application/json'}
    res = requests.post(endpoint, headers=headers, json=data)
    return res.json()


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
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
    return input_sentences[: batch_size]


def test_hf_model(model):
    if model not in hf_model_spec:
        raise ValueError(f"{args.model} is not one of the supporting models {list(hf_model_spec.keys())}")
    spec = hf_model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for batch_size in spec["batch_size"]:
        for seq_length in spec["seq_length"]:
            req = {"inputs": batch_generation(batch_size)}
            params = {"min_length": seq_length, "max_length": seq_length}
            req["parameters"] = params
            res = send_json(req)
            logging.info(f"res {res}")
            result = [item[0]['generated_text'] for item in res]
            assert len(result) == batch_size
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


def test_ds_raw_model(model):
    if model not in ds_raw_model_spec:
        raise ValueError(f"{args.model} is not one of the supporting models {list(ds_raw_model_spec.keys())}")
    spec = ds_raw_model_spec[args.model]
    for batch_size in spec["batch_size"]:
        for seq_length in spec["seq_length"]:
            req = {"batch_size": batch_size, "text_length": seq_length, "use_pipeline": spec["use_pipeline"]}
            logging.info(f"req: {req}")
            res = send_json(req)
            logging.info(f"res: {res}")
            assert len(res["outputs"]) == batch_size
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


supported_handler = {'deepspeed': None, 'huggingface': test_hf_model, "deepspeed_raw": test_ds_raw_model}

if __name__ == '__main__':
    args = parser.parse_args()
    if args.handler not in supported_handler:
        raise ValueError(f"{args.handler} is not one of the supporting handler {list(supported_handler.keys())}")
    supported_handler[args.handler](args.model)
