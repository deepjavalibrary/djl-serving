import urllib.request
import json

dataset_urls = {
    "gsm8k":
    "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/gsm8k.jsonl",
    "humaneval":
    "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/humaneval.jsonl",
    "mbpp":
    "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/mbpp.jsonl",
    "mtbench":
    "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/mt_bench.jsonl",
    "openorca":
    "https://raw.githubusercontent.com/ymwangg/vllm-test/main/dataset/openorca_curated_750.jsonl"
}

parameters = {
    "gsm8k": {
        "temperature": 0.0,
        "stop": ["Question:", "</s>", "<|im_end|>"],
        "max_num_tokens": 256
    },
    "humaneval": {
        "temperature":
        0.2,
        "stop": [
            "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```",
            "<file_sep>"
        ],
        "max_num_tokens":
        512
    },
    "mbpp": {
        "temperature":
        0.1,
        "stop":
        ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
        "max_num_tokens":
        512
    },
    "mtbench": {
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 40,
        "max_num_tokens": 512
    },
    "openorca": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_num_tokens": 256
    }
}


def gsm8k_parsing(row: dict, parameters):
    return {"inputs": row["prompt"], "parameters": parameters, "stream": True}


def humaneval_parsing(row: dict, parameters):
    return {"inputs": row["prompt"], "parameters": parameters, "stream": True}


def mbpp_parsing(row: dict, parameters):
    return {"inputs": row["prompt"], "parameters": parameters, "stream": True}


def mtbench_parsing(row: dict, parameters):
    return {"inputs": row["prompt"], "parameters": parameters, "stream": True}


def openorca_parsing(row: dict, parameters):
    return {"inputs": row["prompt"], "parameters": parameters, "stream": True}


parsing = {
    "gsm8k": gsm8k_parsing,
    "humaneval": humaneval_parsing,
    "mbpp": mbpp_parsing,
    "mtbench": mtbench_parsing,
    "openorca": openorca_parsing
}


def build_djl_serving_request(dataset):
    with open(f"{dataset}_djlserving.jsonl", "w") as f:
        for line in urllib.request.urlopen(dataset_urls[dataset]):
            line = json.loads(line.decode('utf-8'))
            request_line = parsing[dataset](line, parameters[dataset])
            f.writelines(f"{json.dumps(request_line)}\n")


build_djl_serving_request("gsm8k")
build_djl_serving_request("humaneval")
build_djl_serving_request("mbpp")
build_djl_serving_request("mtbench")
build_djl_serving_request("openorca")
