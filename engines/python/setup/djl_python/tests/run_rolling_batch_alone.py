"""
Run the below code like

pip install git+https://github.com/deepjavalibrary/djl-serving.git#subdirectory=engines/python/setup

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  run_rolling_batch_alone.py openlm-research/open_llama_7b_v2 -rb lmi-dist
"""
import argparse
import logging
import json
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
                       properties: dict = None):
    rolling_batch_type = rolling_batch_type.lower()
    if not properties:
        properties = {"tensor_parallel_degree": 1, "trust_remote_code": True}
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
    if args.rollingbatch == "vllm" or args.rollingbatch == "lmi-dist":
        dist.init_process_group("nccl")
    properties = None
    if args.properties:
        properties = json.loads(args.properties)
    batcher = init_rolling_batch(args.rollingbatch, args.model_id, properties)
    input_data = ["Deep Learning is"]
    parameters = [{}]
    result = batcher.inference(input_data, parameters)
    print_rank0(result)
