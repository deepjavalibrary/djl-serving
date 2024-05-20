import sys
import requests
import argparse
import subprocess as sp
import logging
import re
import os
import math
import json
import shutil
from random import randrange

logging.basicConfig(level=logging.INFO)


def get_model_name():
    endpoint = f"http://127.0.0.1:8080/models"
    res = requests.get(endpoint).json()
    return res["models"][0]["modelName"]


onnx_model_spec = {
    "bge-base-en-v1.5": {
        "max_memory_per_gpu": [1.0, 1.0, 1.0, 1.0],
        "batch_size": [1, 2, 4, 8],
    },
    "bge-reranker": {
        "max_memory_per_gpu": [2.0, 2.0, 2.0, 2.0],
        "batch_size": [1, 2, 4, 8],
    }
}


def check_worker_number(desired):
    model_name = get_model_name()
    endpoint = f"http://127.0.0.1:8080/models/{model_name}"
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
    endpoint = f"http://127.0.0.1:8080/invocations"
    resp = requests.post(endpoint, headers=headers, json=data)

    if resp.status_code >= 300:
        logging.exception(f"HTTP error: {resp}")
        raise ValueError("Failed to send reqeust to model server")
    return resp


def find_awscurl():
    command = "./awscurl -h"
    try:
        sp.check_output(command, shell=True)
    except sp.CalledProcessError:
        logging.info("Downloading awscurl...")
        command = "wget https://publish.djl.ai/awscurl/awscurl && chmod +x awscurl"
        sp.call(command, shell=True)


def awscurl_run(data, concurrency, num_run=5, dataset=False):
    find_awscurl()
    headers = "Content-type: application/json"
    endpoint = f"http://127.0.0.1:8080/invocations"
    if dataset:
        dataset_dir = os.path.join(os.path.curdir, "dataset")
        os.mkdir(dataset_dir)
        for i, d in enumerate(data):
            with open(os.path.join(dataset_dir, f"prompt{i}.txt"), "w") as f:
                f.write(json.dumps(d))
        command_data = f"--dataset {dataset_dir}"
    else:
        json_data = json.dumps(data)
        command_data = f"-d '{json_data}'"
    command = (f"./awscurl -c {concurrency} "
               f"-N {num_run} -X POST {endpoint} --connect-timeout 120 "
               f"-H {headers} {command_data} -P")
    logging.info(f"Running command {command}")
    sp.call(command, shell=True)
    if dataset:
        shutil.rmtree(dataset_dir)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]

    def convert_str_to_mem_used_gb(mem_free_info):
        return float(mem_free_info.split()[0]) / 1024.0

    return [
        convert_str_to_mem_used_gb(x) for i, x in enumerate(memory_free_info)
    ]


def validate_memory_usage(expected_memory_limit):
    used_memory_per_gpu = get_gpu_memory()
    logging.info(f"Used memory per GPU: {used_memory_per_gpu}")
    if any(x > expected_memory_limit for x in used_memory_per_gpu):
        raise AssertionError(f"Memory usage is too high!"
                             f"Used Memory:{used_memory_per_gpu}"
                             f"Expected Upper Limit:{expected_memory_limit}")


def fake_tokenizer(prompt, in_tokens):
    tokenized = re.findall(r"[\w']+|[.,!?;]", prompt)
    index_pointer = 0
    token_count = 0
    for token in tokenized:
        target = token[-1]
        index_pointer = prompt.find(target, index_pointer) + 1
        token_count += 1
        if token_count == in_tokens:
            break
    return prompt[:index_pointer]


def prompt_generation(in_tokens):
    with open(os.path.join(os.getcwd(), 'prompts.txt')) as f:
        result = '\n'.join(f.readlines())
    rot = result.find('. ', randrange(len(result))) + 2
    result = result[rot:] + result[:rot]

    return fake_tokenizer(result, in_tokens)


def batch_generation(batch_size):
    if args.in_tokens:
        input_sentences = [
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens),
            prompt_generation(args.in_tokens)
        ]
    else:
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


def test_onnx_model(model):
    if model not in onnx_model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(onnx_model_spec.keys())}"
        )
    spec = onnx_model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for i, batch_size in enumerate(spec["batch_size"]):
        req = {"inputs": batch_generation(batch_size)}
        logging.info(f"req {req}")
        res = send_json(req).json()
        logging.info(f"res: {res}")
        assert len(res) == batch_size
        if "max_memory_per_gpu" in spec:
            validate_memory_usage(spec["max_memory_per_gpu"][i])

        # awscurl little benchmark phase
        logging.info(f"Little benchmark: concurrency {batch_size}")
        awscurl_run(req, batch_size)


supported_engine = {
    'onnxruntime': test_onnx_model,
}


def run(raw_args):
    parser = argparse.ArgumentParser(description="Build the LLM configs")
    parser.add_argument('engine',
                        type=str,
                        choices=['onnxruntime'],
                        help='The engine used for inference')
    parser.add_argument('model',
                        type=str,
                        help='model that works with certain engine')
    parser.add_argument("--dtype",
                        required=False,
                        type=str,
                        help="The model data type")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        help="The batch size of inference requests")
    parser.add_argument("--in_tokens",
                        required=False,
                        type=int,
                        help="The sequence length for input tokens")
    parser.add_argument("--count",
                        required=False,
                        type=int,
                        help="Number of requests sent")
    parser.add_argument("--cpu_memory",
                        required=False,
                        default=0,
                        type=int,
                        help="CPU Memory footprint")
    global args
    args = parser.parse_args(args=raw_args)

    if args.engine not in supported_engine:
        raise ValueError(
            f"{args.engine} is not one of the supporting engine {list(supported_engine.keys())}"
        )
    supported_engine[args.engine](args.model)

    if args.engine == "onnxruntime":
        test_onnx_model(args.model)


if __name__ == "__main__":
    run(sys.argv[1:])
