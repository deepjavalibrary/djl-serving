

# LMI NeuronX ahead-of-time compilation of models tutorial

## Overview

With LMI NeuronX container, you can run on the fly compilation of different LLM architecture models and then load the model and run inference. However, for larger models like Llama-2 70B models, compilation of models takes approximately 70 minutes and more. So, we recommend users to perform ahead-of-time compilation for these larger models and other model architectures to avoid this compilation overhead.

The goal of this document is for the user to be able to:

1. Convert the HuggingFace model to LMI NeuronX model format.
2. Upload the model to S3 so you could use it for inference.

## Supported Runtime-Compilation architectures

* LLaMA (since LMI V7 0.25.0)
* Mistral (since LMI V8 0.26.0)
* GPT-NeoX (since LMI V5 0.23.0)
* GPT-J (since LMI V5 0.23.0)
* GPT2 (since LMI V5 0.23.0)
* OPT (since LMI V5 0.23.0)
* Bloom (since LMI V5 0.23.0)

## Step-by-step tutorial

In this tutorial, we will be converting the Llama2-70b model to LMI NeuronX model format on inf2.48xlarge.

### Step 1: Choose your instance

To do the model compilation you need to use the instance that has the same Neuron architecture that will be used for your inference deployment (e.g. model compiled on a inf2 instance can be deployed only on inf2 instances).

For this example we will be using the Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) on an inf2.48xlarge - and will assume that we are working from the home directory there. You can do this process on a different AMI, but it may require some changes to the directories pointed to when using the docker image.

### Step 2: Pull the docker image

Refer [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers) for the latest LMI NeuronX DLC and pull the image.

For example:

```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# Download docker image
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.26.0-neuronx-sdk2.16.0

```

### Step 3: Set the environment variables:

These below configurations helps you configure the inference optimizations parameters. You can check all the configurations of LMI NeuronX handler [in our docs](../user_guides/tnx_user_guide.md#advanced-transformers-neuronx-configurations).
You will save these as a `serving.properties` text file that will define the models compilation configuration.

___serving.properties___
```
engine=Python
option.model_id={{huggingface_model_id/s3url/directory}}
option.entryPoint=djl_python.transformers_neuronx
option.tensor_parallel_degree=24
option.n_positions=512
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.enable_mixed_precision_accumulation=true
option.model_loading_timeout=12000
option.save_mp_checkpoint_path=/opt/ml/input/data/training/partition-test
```

### Step 4: Setup directories and model artifacts for compilation

In this tutorial using Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) on an inf2.48xlarge and assume we are working the default home directory `~/`. If you are using a HuggingFace `model_id` or s3 url for your model then you can move on to the next step.

For downloading a model to a local directory we will be using the following script (model and directory can be updated for your use case):

___download.py___
```python
from huggingface_hub import snapshot_download
from pathlib import Path

# - This will download the model into the ./model directory where ever the script is running
local_model_path = Path("./model")
local_model_path.mkdir(exist_ok=True)
model_name = "meta-llama/Llama-2-70b-hf"
# Only download safetensors checkpoint files
allow_patterns = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.model", "*.tiktoken"]

# - Leverage the snapshot library to download the model since the model is stored in repository using LFS
snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
    token="hf_YOUR_TOKEN_VALUE",
)
```

We will run the `download.py` script defined above in order to download a local copy of the model into the `~/model` directory.

```bash
pip install huggingface_hub pathlib
python3 download.py
```

After the download completes the `~/model` directory will contain the model artifacts that we can use for compilation.

### Step 5: Run the ahead-of-time compilation

Run the container and run the model partitioning script. 

The command below assumes that we are using a `serving.properties` in the `~/` directory and that we will be downloading the model artifacts from s3. If you need to download the model from HuggingFace using an access token you would add the following line to the `docker run` command:

```
  -e HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_VALUE" \
```

Tutorial `docker run` for our example model compilation.

```bash
docker run -t --rm --network=host \
  -v /home/ubuntu/:/opt/ml/input/data/training \
  --device /dev/neuron0 \
  --device /dev/neuron1 \
  --device /dev/neuron2 \
  --device /dev/neuron3 \
  --device /dev/neuron4 \
  --device /dev/neuron5 \
  --device /dev/neuron6 \
  --device /dev/neuron7 \
  --device /dev/neuron8 \
  --device /dev/neuron9 \
  --device /dev/neuron10 \
  --device /dev/neuron11 \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.26.0-neuronx-sdk2.16.0 \
  partition --model-dir /opt/ml/input/data/training --skip-copy
```

### Step 6: Upload the compiled model to S3

To upload the model to S3, run the command below. If you have moved the model or changed the above scripts you will want to update the command below to reflect your changes.

```bash
aws s3 cp ~/partition-test s3://YOUR_S3_FOLDER_NAME/ --recursive
```

Note: After uploading model artifacts to s3, you can just update the model_id(in `serving.properties`) to the newly created s3 url with compiled model artifacts and use the same `serving.properties` when deploying on SageMaker. 
Here, you can check the [tutorial](https://github.com/deepjavalibrary/djl-demo/blob/master/aws/sagemaker/large-model-inference/sample-llm/tnx_rollingbatch_deploy_llama_70b.ipynb) on how to run inference using LMI NeuronX DLC. Below snippet shows example updated model_id.

___serving.properties___
````
engine=Python
option.model_id=s3://YOUR_S3_FOLDER_NAME/
option.entryPoint=djl_python.transformers_neuronx
option.tensor_parallel_degree=24
option.n_positions=512
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.enable_mixed_precision_accumulation=true
option.model_loading_timeout=3600
```
