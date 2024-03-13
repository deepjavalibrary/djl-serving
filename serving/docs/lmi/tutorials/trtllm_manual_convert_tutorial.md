# TensorRT-LLM manual compilation of models tutorial

## Overview

With LMI TensorRT-LLM container, you can run manual compilation of some models and quantization that LMI not supported for JIT and then load the model and run inference.

The goal of this document is for the user to be able to:

1. Convert any TensorRT-LLM supported model to a model format that LMI can load and run inference.
2. Upload the model to S3 so you could use it for runtime.

## Step by step tutorial

In this tutorial, we will be converting the baichuan model to TensorRT-LLM model format on p4d.24xlarge.

### Step 1: Choose your instance

To do the model compilation you need to use the instance that has **the same GPU architecture** that will be used for your inference deployment (e.g model compiled on a g5 instance can be deployed only on g5 instances).

### Step 2: Pull the docker image

Refer [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers) for the latest TensorRT-LLM DLC and pull the image.

For example:

```
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-tensorrtllm0.7.1-cu122
```

You can also pull the container from DockerHub:

```
docker pull deepjavalibrary/djl-serving:0.26.0-tensorrt-llm
```

### Step 3: Login the container and prepare the environment

You need to manually login into the container to proceed for conversion

```
docker run -it --runtime=nvidia --gpus all \
--shm-size 12g \
deepjavalibrary/djl-serving:0.26.0-tensorrt-llm \
/bin/bash
```

Then clone the TensorRT-LLM repository inside the container for manual conversion.
Please make sure the version of the TensorRT-LLM need to match the one installed on the system.

You can check it with

```
pip show tensorrt-llm
# Output
# Name: tensorrt-llm
# Version: 0.7.1
```

Then just clone the TensorRT-LLM Triton backend for model preparation. If the version is 0.5.0, then you need to checkout the tag for that version (v0.5.0).
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend -b v0.5.0
cd tensorrtllm_backend && rm -rf tensorrt_llm
git clone https://github.com/NVIDIA/TensorRT-LLM -b v0.5.0 tensorrt_llm
```


### Step 4: Build TensorRT-LLM compatible model

The following work we do is very similar to Triton Server model preparation. You can find more information on the [official preparation doc](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0?tab=readme-ov-file#prepare-tensorrt-llm-engines).
Our environment is pre-configured all necessary packages, so we don't need to pip install anything.

Here we just need to

```
cd tensorrt_llm/examples/baichuan
# or just tensorrt_llm/examples/<model you like>
```

Checking the `README.md` we will find the following instruction. We just need to add a few more parameter to build ready-to-go model:

```
# Build the Baichuan V1 13B model using a single GPU and FP16.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --world_size 2 \
                --output_dir baichuan_v1_13b/trt_engines/fp16/2-gpu/ \
                --use_inflight_batching
```

We added parameter called `--use_inflight_batching`, this will help us to do continuous batching with LMI.

Generally speaking, we would recommend to turn those flags on if the conversion script support:
- `--enable_context_fmha` to speed up inference with deep optimization
- `--remove_input_padding` to not allow padding and reduce memory cost
- `--parallel_build` do build in parallel to speed up

After the conversion, you will find the model in `baichuan_v1_13b/trt_engines/fp16/2-gpu/`,
you can change the output directory with `--output_dir`. Also remember we are using world size 2,
this means we are sharding the model on 2 gpus. Tt will be the same as `tensor_parallel_degree` we specified in LMI.

After the conversion, let's make the model format ready:

```
# cd to tensorrtllm_backend folder level
cd ../../../
mkdir -p triton_model_repo/tensorrt_llm/
cp -r all_models/inflight_batcher_llm/tensorrt_llm/* triton_model_repo/tensorrt_llm/
# copy the converted model to the repo
cp tensorrt_llm/examples/baichuan/baichuan_v1_13b/trt_engines/fp16/2-gpu/* triton_model_repo/tensorrt_llm/1
```

Then, let's configure the model settings. Here we are using a tool called template filler that are available in the repo:

```
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt \
"enable_trt_overlap:False,batch_scheduler_policy:max_utilization,kv_cache_free_gpu_mem_fraction:0.95,max_num_sequences:64"
```

This tool will help us fill in most of the config we need. Finally, we need to set the following property manually:

```
vi triton_model_repo/tensorrt_llm/config.pbtxt
'''
model_transaction_policy {
  decoupled: True
}
'''
```

Change the value to `True`. 

We also need to delete/change the value the parameters we didn't define a value like:

```
parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: {
    string_value: "${max_tokens_in_paged_kv_cache}"
  }
}
```

Here is a table contains TRTLLM supported parameters with our suggestions. (up to 0.6.1)

| Name                           | Settings                | Meaning                                                                                                                                                                        |
|--------------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| enable _trt_overlap            | False                   | Allow TRT to overlap calculation, this helped model perform better when using higher concurrency. But during our test, we found that setting this to False helped performance. |
| max_tokens_in_paged_kv_cache   | Remove this             | For PagedAttention, TensorRT LLM will calculate this value based on batch size and model compilation settings.                                                                 |
| batch_scheduler_policy         | max_utilization         | max GPU utlization for most of the case                                                                                                                                        |
| kv_cache_free_gpu_mem_fraction | 0.95                    | The percentage of GPU memory the model can us for KV Cache management. Here we are setting an agressive number to 0.95 to max out GPU                                          |
| max_num_sequences              | 64                      | The maximum concurrent batch size you can send as inputs                                                                                                                       |
| max_beam_width                 | 1                       | use beam size 1 for generation. Beam size > 1 is not supported currently for LMI                                                                                               |
| gpt_model_type                 | inflight_fused_batching | Used continuous batching mechanism                                                                                                                                             |
| gpt_model_path                 | /tmp                    | LMI will change this value in runtime to correct the model path after download. Just leave it there and not remove it                                                          |
| max_kv_cache_length            | Remove this             | For PagedAttention, TensorRT LLM will calculate this value based on batch size and model compilation settings.                                                                 |
| exclude_input_in_output        | True                    | This will helps to follow the same way LMI generating as the result. Remove the prefix (input)     


(0.6.1) Also need to remove some configs that are used for dynamic batching:

```
dynamic_batching {
    preferred_batch_size: [ ${triton_max_batch_size} ]
    max_queue_delay_microseconds: ${max_queue_delay_microseconds}
}
```


### Step 5: Prepare HuggingFace configs and tokenizers

Besides the standard Triton components, we also need to add the tokenizer stuff from HuggingFace to here.
Create a file called `build_tokenizer.py` with the following contents:

```python
from huggingface_hub import snapshot_download
from pathlib import Path

local_model_path = Path("./triton_model_repo/tensorrt_llm/")
model_name = "baichuan-inc/Baichuan-13B-Chat"
# Only download pytorch checkpoint files
allow_patterns = ["*.json", "*.txt", "*.model"]

snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    allow_patterns=allow_patterns,
    local_dir_use_symlinks=False
)
```

Run with:

```bash
python build_tokenizer.py
```

It will download all necessary components to the folder.

Finally, let's see what's inside:

```bash
ls triton_model_repo/tensorrt_llm/
# 1  config.json  config.pbtxt  generation_config.json  pytorch_model.bin.index.json  requirements.txt  special_tokens_map.json  tokenizer.model  tokenizer_config.json
ls triton_model_repo/tensorrt_llm/1/
# baichuan_float16_tp2_rank0.engine  baichuan_float16_tp2_rank1.engine  config.json  model.cache
```

### Step 6: Upload the compiled model to S3

Prepare necessary credentials for your models and do upload:

```
aws s3 sync triton_model_repo/tensorrt_llm/ s3://lmi-llm/trtllm/0.5.0/baichuan-13b-tp2/baichuan-13b-chat/
```

Note: We always need to create two folders here to load the model. Saying if you store like the followings:

```
aws s3 sync triton_model_repo/tensorrt_llm/ s3://<some-bucket>/...<some_folders>../folder1/folder2/
```

The S3 url used in LMI to load the model will be
```
s3://<some-bucket>/...<some_folders>../folder1/
```

and in our case is:

```
s3://lmi-llm/trtllm/0.5.0/baichuan-13b-tp2/
```

Check the file is there

```
aws s3 ls s3://lmi-llm/trtllm/0.5.0/baichuan-13b-tp2/baichuan-13b-chat/
                           PRE 1/
2023-12-19 01:58:03        733 config.json
2023-12-19 01:58:03       4425 config.pbtxt
2023-12-19 01:58:02        284 generation_config.json
2023-12-19 01:58:02      23274 pytorch_model.bin.index.json
2023-12-19 01:58:03         56 requirements.txt
2023-12-19 01:58:03        544 special_tokens_map.json
2023-12-19 01:58:03    1136765 tokenizer.model
2023-12-19 01:58:03        954 tokenizer_config.json
```

## Load on SageMaker LMI container

Finally, you can use one of the following configuration to load your model on SageMaker:

 ### 1. Environment variables:
```
OPTION_MODEL_ID=s3://lmi-llm/trtllm/0.5.0/baichuan-13b-tp2/
OPTION_TENSOR_PARALLEL_DEGREE=2
OPTION_MAX_ROLLING_BATCH_SIZE=64
```

### 2. `serving.properties`:

```
engine=MPI
option.model_id=s3://lmi-llm/trtllm/0.5.0/baichuan-13b-tp2/
option.tensor_parallel_degree=2
option.max_rolling_batch_size=64
```

### 3. extracted model artifacts:

`serving.properties`:
```
engine=MPI
option.rolling_batch=trtllm
option.dtype=fp16
option.tensor_parallel_degree=2
```

Artifacts need to be in the following structure:

Mount should be to `/opt/ml/model/`
```
├── serving.properties
└── tensorrt_llm
    ├── 1
    │   ├── baichuan_float16_tp2_rank0.engine
    │   ├── baichuan_float16_tp2_rank1.engine
    │   ├── config.json
    │   └── model.cache
    ├── config.json
    ├── config.pbtxt
    ├── configuration_baichuan.py
    ├── generation_config.json
    ├── pytorch_model.bin.index.json
    ├── requirements.txt
    ├── special_tokens_map.json
    ├── tokenization_baichuan.py
    ├── tokenizer_config.json
    └── tokenizer.model
```

`config.pbtxt`:
Make sure to update `gpt_model_path` to the correct path including parent folder name (`/opt/ml/model/tensorrt_llm/1`)

```
parameters: {
  key: "gpt_model_path"
  value: {
    string_value: "/opt/ml/model/tensorrt_llm/1"
  }
}
```

