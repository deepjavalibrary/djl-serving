import torch

import os, sys

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.10/dist-packages/lmi_dist")

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
from djl_python.tests.rolling_batch_test_scripts.generator import Generator, print_rank0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 0 if torch.cuda.is_available() else 0
"""
option.model_id=TheBloke/Llama-2-13B-Chat-fp16
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.model_loading_timeout=7200
option.max_rolling_batch_prefill_tokens=36080
"""
properties = {
    "mpi_mode": "true",
    "tensor_parallel_degree": 1,
    "dtype": "fp16",
    "max_rolling_batch_size": 28,
    "model_loading_timeout": 3600,
    "max_rolling_batch_prefill_tokens": 1000,
    "paged_attention": "True"
}

# model_id = "TheBloke/Llama-2-13B-Chat-fp16"  # multi gpu; 7,236 MiBx4
# model_id = "openlm-research/open_llama_7b_v2"

# model_id = "huggyllama/llama-7b"  # 9,542MiB / 23,028MiB;
# model_id = "JackFram/llama-160m"  #   844MiB / 23,028MiB;

# model_id = "bigscience/bloom-560m"  # OOM on a single gpu and not sharded on multi gpu
# model_id = "gpt2"
# model_id = "facebook/opt-125m"

model_id = "TheBloke/Llama-2-7B-Chat-fp16"  # 14,114MiB / 23,028MiB
draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"  #  2,710MiB / 23,028MiB
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
# weight model.layers.0.self_attn.rotary_emb.inv_freq does not exist
# model_id = "TinyLlama/TinyLlama-1.1B-python-v0.1"
# model_id = "codellama/CodeLlama-7b-hf"  # 14,054MiB / 23028MiB;
# draft_model_id = None
properties['spec_length'] = 3

# ===================== lmi ============================
device = int(os.environ.get("RANK", 0))
# device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

properties["model_id"] = model_id
properties["draft_model_id"] = draft_model_id
properties["device"] = device
rolling_batch = LmiDistRollingBatch(model_id, properties)

gen = Generator(rolling_batch=rolling_batch)

print('========== init inference ===========')
input_str1 = [
    "Hello, my name is",  # 6
    "The president of the United States is",  # 8
    "The capital of France is",  # 6
    "The future of AI is"
]  # 7
params1 = [{
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.001
}, {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.001
}, {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.001
}, {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.001
}]
# params1 = [{"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1}]

gen.step(step=10, input_str_delta=input_str1, params_delta=params1)

for _ in range(1):
    print('========== inference1 ===========')
    input_str_delta = [
        "Hello, my name is Hello, my name is Hello, my name is Hello, my name is",  # 21
        "Hello, my name is Hello, my name is Hello, my name is"
    ]  # 16

    params_delta = [{
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.001
    }, {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.001
    }]
    # params_delta = [{"max_new_tokens":100, "do_sample":True, "temperature":1},
    #                 {"max_new_tokens":100, "do_sample":True, "temperature":1}]

    gen.step(step=10,
             input_str_delta=input_str_delta,
             params_delta=params_delta)

print('========== inference_infty ===========')
gen.step(step=500)
for req_id, out in gen.output_all.items():
    print_rank0(
        f"\n====req_id: {req_id}=====\n{gen.input_all[req_id][0] + ''.join(out)}\n"
    )
