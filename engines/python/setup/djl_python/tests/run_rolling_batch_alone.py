"""
Run the below code like

pip install git+https://github.com/deepjavalibrary/djl-serving.git#subdirectory=engines/python/setup

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  run_rolling_batch_alone.py openlm-research/open_llama_7b_v2 -rb lmi-dist
"""
import argparse
import logging
import json
import random
from typing import List

import torch.distributed as dist


def get_rolling_batch_class_from_str(rolling_batch_type: str):
    if rolling_batch_type == "scheduler":
        from djl_python.rolling_batch import SchedulerRollingBatch
        return SchedulerRollingBatch
    elif rolling_batch_type == "lmi-dist":
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        return LmiDistRollingBatch
    elif rolling_batch_type == "vllm":
        from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
        logging.warning(
            "vLLM rolling batcher is experimental, use with caution")
        return VLLMRollingBatch
    raise ValueError(f"Invalid rolling batch type: {rolling_batch_type}")


def init_rolling_batch(rolling_batch_type: str,
                       model_id: str,
                       properties: dict):
    rolling_batch_type = rolling_batch_type.lower()
    device = 0
    if dist.is_initialized():
        device = dist.get_rank()
        properties["tensor_parallel_degree"] = dist.get_world_size()
    rolling_batcher_cls = get_rolling_batch_class_from_str(rolling_batch_type)
    return rolling_batcher_cls(model_id, device, properties, **properties)


def print_rank0(content):
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    if rank == 0:
        print(content)


data_collector = []


def collect_data(result):
    done_requests_indices = []
    for idx, item in enumerate(result):
        if len(data_collector) <= idx:
            data_collector.append(item["data"])
        else:
            data_collector[idx] += item["data"]
        if item['last']:
            done_requests_indices.append(idx)
    for idx in sorted(done_requests_indices, reverse=True):
        value = data_collector.pop(idx)
        print_rank0(f"\nFinish one request: {value}\n")
    return done_requests_indices


def build_request(sample_prompt: List,
                  sample_param: List,
                  batch_size,
                  shuffle=False):
    sample_prompt = sample_prompt.copy()
    sample_param = sample_param.copy()
    while len(sample_prompt) < batch_size:
        sample_prompt.extend(sample_prompt)
        sample_param.extend(sample_param)
    if shuffle:
        temp = list(zip(sample_prompt, sample_param))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        sample_prompt, sample_param = list(res1), list(res2)
    return sample_prompt[:batch_size], sample_param[:batch_size]


def simulator(batcher,
              sample_prompt: str or List,
              sample_param: dict or List,
              merge_batch_size: List,
              wait_step: int or list = 1,
              shuffle=False):
    if isinstance(sample_prompt, str):
        sample_prompt = [sample_prompt]
    if isinstance(sample_param, dict):
        sample_param = [sample_param] * len(sample_prompt)
    if isinstance(wait_step, int):
        wait_step = [wait_step] * len(merge_batch_size)
    assert len(sample_prompt) == len(sample_param)
    assert len(merge_batch_size) == len(wait_step)
    current_prompt = []
    current_param = []
    for batch_size, step in zip(merge_batch_size, wait_step):
        new_prompt, new_param = build_request(sample_prompt, sample_param,
                                              batch_size, shuffle)
        current_prompt.extend(new_prompt)
        current_param.extend(new_param)
        for i in range(step):
            if len(current_prompt) == 0:
                break
            generated_tokens = batcher.inference(current_prompt, current_param)
            finished_indices = collect_data(generated_tokens)
            for idx in sorted(finished_indices, reverse=True):
                current_prompt.pop(idx)
                current_param.pop(idx)
    while len(current_prompt) > 0:
        generated_tokens = batcher.inference(current_prompt, current_param)
        finished_indices = collect_data(generated_tokens)
        for idx in sorted(finished_indices, reverse=True):
            current_prompt.pop(idx)
            current_param.pop(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id",
                        type=str,
                        help="the model id need to run with")
    parser.add_argument("-rb",
                        "--rollingbatch",
                        type=str,
                        choices=["scheduler", "vllm", "lmi-dist"])
    parser.add_argument("--properties",
                        type=str,
                        required=False,
                        help="The properties needed for the rolling batcher")
    args = parser.parse_args()
    properties = None
    if args.properties:
        properties = json.loads(args.properties)
    else:
        properties = {"tensor_parallel_degree": 1, "trust_remote_code": True, "engine": "Python"}
    if args.rollingbatch == "lmi-dist":
        dist.init_process_group("nccl")
        properties["engine"] = "MPI"
    batcher = init_rolling_batch(args.rollingbatch, args.model_id, properties)
    simulator(batcher, "write a program that can sum two number in python", {
        "max_new_tokens": 256,
        "do_sample": True
    }, [1, 1, 1, 1], 1)
