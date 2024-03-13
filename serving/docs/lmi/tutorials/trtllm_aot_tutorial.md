# TensorRT-LLM ahead-of-time compilation of models tutorial

## Overview

With LMI TensorRT-LLM container, you can run on the fly compilation of different LLM architecture models and then load the model and run inference. 
However, for larger models like Falcon 40B and Llama-2 70B models, compilation of models takes approximately 7 minutes and more. 
So, we recommend users to perform ahead-of-time compilation for these larger models and other model architectures to avoid this compilation overhead.

The goal of this document is for the user to be able to:

1. Convert the HuggingFace model to TensorRT-LLM model format.
2. Upload the model to S3 so you could use it for inference. 

## Supported JIT architecture

- LLaMA (since LMI V7 0.25.0)
- Falcon (since LMI V7 0.25.0)
- InternLM (since LMI V8 0.26.0)
- Baichuan (since LMI V8 0.26.0)
- ChatGLM (since LMI V8 0.26.0)
- GPT-J (since LMI V8 0.26.0)
- Mistral (since LMI V8 0.26.0)
- Mixtral (since LMI V8 0.26.0)
- Qwen (since LMI V8 0.26.0)
- GPT2/SantaCoder/StarCoder (since LMI V8 0.26.0)

For model that are not listed here, you can use [this tutorial](trtllm_manual_convert_tutorial.md) instead to prepare model manually.

## Step by step tutorial

In this tutorial, we will be converting the Llama2-70b model to TensorRT-LLM model format on p4d.24xlarge.

### Step 1: Choose your instance

To do the model compilation you need to use the instance that has **the same GPU architecture** that will be used for your inference deployment (e.g model compiled on a g5 instance can be deployed only on g5 instances).

### Step 2: Pull the docker image

Refer [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers) for the latest TensorRT-LLM DLC and pull the image.

For example:

```
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-tensorrtllm0.7.1-cu122
```

### Step 3: Set the environment variables:

These below configurations helps you configure the inference optimizations parameters. You can check all the configurations of TensorRT-LLM LMI handler  [in our docs](../user_guides/trt_llm_user_guide.md#advanced-tensorrt-llm-configurations). 

```
OPTION_MODEL_ID={{s3url}}
OPTION_TENSOR_PARALLEL_DEGREE=8
OPTION_MAX_ROLLING_BATCH_SIZE=128
OPTION_DTYPE=fp16
```

You can also use serving.properties instead of these above environment variables.

```
%%writefile serving.properties
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=128
option.dtype=fp16
```

### Step 4: Create local model directory

In your current working directory, create `<model_repo_dir>` (give your own custom name) where you want to store the compiled model artifacts.

```
MODEL_REPO_DIR=$PWD/<model_repo_dir>
mkdir -p $MODEL_REPO_DIR
```

In the next step we will map `$MODEL_REPO_DIR` to a volume inside container and pass that path in the container to the partition script.

### Step 5: Run the ahead of time compilation

Run the container and run the model partitioning script. 
Remember to map all the environment variables you set in step 2 to inside the container by adding `-e YOUR_OPTION_NAME=$YOUR_OPTION_NAME` to the docker run command. 
In the below example, the model artifacts will be saved to `$MODEL_REPO_DIR` created in the above step. (Also, if you are using `serving.properties` file to set the options, there’s no need to pass `OPTION_*` environment variables in the below command.)

```
docker run --runtime=nvidia --gpus all --shm-size 12gb \
-v $MODEL_REPO_DIR:/tmp/trtllm \
-p 8080:8080 \
-e OPTION_MODEL_ID=$OPTION_MODEL_ID \
-e OPTION_TENSOR_PARALLEL_DEGREE=$OPTION_TENSOR_PARALLEL_DEGREE \
-e OPTION_MAX_ROLLING_BATCH_SIZE=$OPTION_MAX_ROLLING_BATCH_SIZE \
-e OPTION_DTYPE=$OPTION_DTYPE \
 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-tensorrtllm0.7.1-cu122 python /opt/djl/partition/trt_llm_partition.py \
--properties_dir $PWD \
--trt_llm_model_repo /tmp/trtllm \
--tensor_parallel_degree $OPTION_TENSOR_PARALLEL_DEGREE
```

Arguments passed to the model compilation and partition script:

* properties_dir - The directory with `serving.properties` file. This will only be used if `serving.properties` file exists otherwise it has no effect if the properties are set by environment variables.
* trt_llm_model_repo - `$MODEL_REPO_DIR` where the model artifacts would be saved.
* tensor_parallel_degree: Should be the same as `$OPTION_TENSOR_PARALLEL_DEGREE`

### Step 6: Upload the compiled model to S3

To upload the model to S3, run the command below, `$MODEL_REPO_DIR` is the same path you set in step 3 and replace YOUR_S3_FOLDER_NAME with whatever you want to name your S3 folder. (Remember to do authentication if needed.)

```
aws s3 cp $MODEL_REPO_DIR s3://YOUR_S3_FOLDER_NAME/ --recursive
```


**Note:**  After uploading model artifacts to s3, you can just update the model_id(env var or in `serving.properties`) to the newly created s3 url with compiled model artifacts and use the same rest of the environment variables or `serving.properties`  when deploying on SageMaker. Here, you can check the [tutorial](https://github.com/deepjavalibrary/djl-demo/blob/master/aws/sagemaker/large-model-inference/sample-llm/trtllm_rollingbatch_deploy_llama_13b.ipynb) on how to run inference using TensorRT-LLM DLC.  Below snippet shows example updated model_id.

```
OPTION_MODEL_ID=s3://YOUR_S3_FOLDER_NAME
OPTION_TENSOR_PARALLEL_DEGREE=8
OPTION_MAX_ROLLING_BATCH_SIZE=128
OPTION_DTYPE=fp16
```
