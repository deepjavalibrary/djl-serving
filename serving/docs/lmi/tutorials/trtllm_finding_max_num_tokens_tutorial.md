# TensorRT-LLM finding max_num_tokens tutorial

From LMI 0.27.0 (TRTLLM 0.8.0), we introduced `max_num_tokens` build option to simplify the compilation of TRTLLM model as well as supporting all kinds of traffic pattern.
This max number of tokens represent the shared tokens across the TRTLLM framework regardless of the batch size or context length.
It is similar to pagedAttention tokens (e.g max_batched_tokens) settings in vLLM. However, this value is controlling the engine cache that has influence to the KVCache tokens itself.

If you set this value in LMI, it is equivalent to the followings:

```
Serving prperties: option.max_num_tokens=50000
ENV: OPTION_MAX_NUM_TOKENS=50000
```

It is the same as:

```
option.max_num_tokens=50000
option.max_input_len=4096
option.max_output_len=4096
option.max_rolling_batch_size=256
```

However, please be aware. Those settings are just to remove the limitation of input/output length and batch size from compilation.
It does not indicate you could do 4096 input/output x 256 batch sizes. It will be based on the actual paged tokens through compilations.

You can also override the value to the number you like. 
For example, Mistral 7B model support 32k context length, you can override input/output length to the value you want. We generally follow the algorithm:

```
option.max_num_tokens=50000
option.max_input_len=max(min(4096,50000),value you override)
option.max_output_len=max(min(4096,50000),value you override)
option.max_rolling_batch_size=max(256,value you override)
```

## Prebuilt table for LLM model under different TP settings

We understand finding the maximum number is difficult, so we precomputed a lookup table for you to find the numbers.
In the future, we will fuse those number into our container and will not ask you to provide one.

### LMI 0.28.0

The following number is tested on the machine with batch size up to 128 and input context up to 3700.

| Model         | Machine  | Tensor Parallel Degree | max number of tokens | 
|---------------|----------|------------------------|----------------------|
| LLaMA 3 8B    | g5.12xl  | 1	                     | 24000                |
| LLaMA 3 8B    | g5.12xl  | 4	                     | 176000               |
| LLaMA 2 7B    | g5.12xl  | 1	                     | 29000                |
| LLaMA 2 7B    | g5.12xl  | 4	                     | 198000               | 
| LLaMA 2 13B   | g5.12xl  | 4                      | 127000               |  
| Gemma 7B      | g5.12xl  | 4                      | 125000               |  
| Gemma 7B      | g5.12xl  | 1                      | 1190                 |  
| Falcon 7B     | g5.12xl  | 1                      | 36000                |  
| Mistral 7B    | g5.12xl  | 1                      | 35000                |  
| Mistral 7B    | g5.12xl  | 4                      | 198000               |  
| LLaMA 2 13B   | g6.12xl  | 4                      | 116000               |
| LLaMA 2 13B   | g5.48xl  | 8                      | 142000               |  
| LLaMA 2 70B   | g5.48xl  | 8                      | 4100                 |  
| LLaMA 3 70B   | g5.48xl  | 8                      | Out of Memory        |  
| Mixtral 8x7B  | g5.48xl  | 8                      | 31000                |  
| Falcon 40B    | g5.48xl  | 8                      | 32000                |  
| CodeLLAMA 34B | g5.48xl  | 8                      | 36000                |
| LLAMA 2 13B   | p4d.24xl | 4                      | 235000               | 
| LLAMA 2 70B   | p4d.24xl | 8                      | 97000                | 
| LLAMA 3 70B   | p4d.24xl | 8                      | 82000                | 
| Mixtral 8x7B  | p4d.24xl | 4                      | 50000                | 
| Mixtral 8x7B  | p4d.24xl | 8                      | 112000               | 
| Falcon 40B    | p4d.24xl | 4                      | 71000                | 
| Mistral 7B    | p4d.24xl | 2                      | 245000               | 
| Mistral 7B    | p4d.24xl | 4                      | 498000               | 
| CodeLLaMA 34B | p4d.24xl | 4                      | 115000               | 
| CodeLLaMA 34B | p4d.24xl | 8                      | 191000               |

### LMI 0.27.0

The following number is tested on the machine with batch size up to 128 and input context up to 3700.

| Model         | Machine  | Tensor Parallel Degree | max number of tokens | 
|---------------|----------|------------------------|----------------------|
| LLaMA 2 7B    | g5.12xl  | 1	                     | 2835                 |
| LLaMA 2 7B    | g5.12xl  | 4	                     | 200000               | 
| LLaMA 2 13B   | g5.12xl  | 4                      | 120000               |  
| LLaMA 2 13B   | g6.12xl  | 4                      | 100000               |  
| Mistral 7B    | g5.12xl  | 1                      | 3287                 |  
| Mistral 7B    | g5.12xl  | 4                      | 200000               | 
| Falcon 7B     | g5.12xl  | 1                      | 2835                 | 
| LLAMA 2 13B   | p4d.24xl | 4                      | 230000               | 
| LLAMA 2 70B   | p4d.24xl | 8                      | 104000               | 
| Mixtral 8x7B  | p4d.24xl | 4                      | 57000                | 
| Falcon 40B    | p4d.24xl | 4                      | 75000                | 
| CodeLLaMA 34B | p4d.24xl | 4                      | 150000               | 
| CodeLLaMA 34B | p4d.24xl | 8                      | 190000               | 

## Find the number with LMI-TRTLLM container solution

Before finding the precise number, you can simply try the following formula by yourself to estimate:

```
max_num_tokens = max_batch_size * max_input_len * alpha
```

Where as the Alpha is a ratio that could be from 0 - 1.0. NVIDIA recommend to set around 0.1 - 0.2 for the best performance.
The reference to the algorithm is [here](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.9.0/docs/source/perf_best_practices.md#maximum-number-of-tokens).

And then try this number with LMI solution and see if it works. 
If not working, go with a binary search approach by reducing halves until you get the number you want.

We also created a simple function to allow you do this with LMI container.

### Step 1: Launch the container

Using your team AWS account and login with DLC account:

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

Then simply do

```
docker run -it --runtime=nvidia --gpus all --shm-size 12gb \
-p 8080:8080 \
-v /opt/dlami/nvme/large_store:/opt/djl/large_store \
-v /opt/dlami/nvme/tmp/.cache:/tmp/.cache \
763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-tensorrtllm0.8.0-cu122 /bin/bash
```

Here we assume you are using g5, g6, p4d, p4de or p5 machine that has NVMe disk available. 
Mounting to that disk will bring you several terabytes of spaces.

### Step 2: Start running with `max_token_finder`

The following step will be executed inside the container.

You can pre-download your huggingface model to `/opt/djl/large_store` or simply fetch from huggingface.
You can create a file named `finder.py`.

```python
from tensorrt_llm_toolkit.utils.utils import max_token_finder

tensor_parallel_degree = 4
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

properties = {
    "model_id": model_id,
    "tensor_parallel_degree": tensor_parallel_degree,
}


model, tp, max_tokens = max_token_finder(properties)
print(f"Summary:\nmodel: {model}\n tp: {tp}\n max_tokens: {max_tokens}")
```

You can override `model_id` to
- local path like `/opt/djl/large_store/<my_downloaded_model_folder>`
- HF model id like `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

`tensor_parallel_degree` means the number of gpu you want to use to parallel computation.
The larger value you set, the bigger max_token the model will be able to support. Finally, just run with:

```
python3 finder.py
```

Internally, we will run a binary search to help you find the number where we set won't cause engine cache OOM.

You can choose the print or save the final calculated `max_tokens` and divide by 2 and finally use that for `option.max_num_tokens` in LMI.
For example, if you get the max_tokens to 50000, set this to 25000 for `option.max_num_tokens`. This will leave some space for pagedKVCache while still enabling engine compute.
