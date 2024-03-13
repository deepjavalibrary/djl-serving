# Table of Contents

- [Overview - Large Model Inference (LMI) Containers](#overview---large-model-inference-lmi-containers)
- [QuickStart](#quickstart)
  - [Using the SageMaker Python SDK](#using-the-sagemaker-python-sdk-to-deploy-your-first-model-with-lmi)
  - [Sample Notebooks](#sample-notebooks)
  - [Deployment Guide](#deployment-guide)
- [Supported LMI Inference Libraries](#supported-lmi-inference-libraries)

# Overview - Large Model Inference (LMI) Containers

LMI containers are a set of high performance Docker Containers purpose built for large language model (LLM) inference. 
With these containers you can leverage high performance open-source inference libraries like [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), 
[DeepSpeed](https://github.com/microsoft/DeepSpeed), [Transformers NeuronX](https://github.com/aws-neuron/transformers-neuronx) to deploy LLMs on [AWS SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html). 
These containers bundle together a model server with open-source inference libraries to deliver an all-in-one LLM serving solution.
We provide quick start notebooks that get you deploying popular open source models in minutes, and advanced guides to maximize performance of your endpoint.

LMI containers provide many features, including:

* Optimized inference performance for popular model architectures like Llama, Bloom, Falcon, T5, Mixtral, and more
* Integration with open source inference libraries like vLLM, TensorRT-LLM, DeepSpeed, and Transformers NeuronX
* Continuous Batching for maximizing throughput at high concurrency
* Token Streaming
* Quantization through AWQ, GPTQ, and SmoothQuant
* Multi GPU inference using Tensor Parallelism
* Serving LoRA fine-tuned models

LMI containers provide these features through integrations with popular inference libraries.
A unified configuration format enables users to easily leverage the latest optimizations and technologies across libraries.
We will refer to each of these libraries as `backends` throughout the documentation. 
The term backend refers to a combination of Engine (LMI uses the Python and MPI Engines) and inference library.
You can learn more about the components of LMI [here](deployment_guide/README.md#components-of-lmi).

## QuickStart

### Using the SageMaker Python SDK to deploy your first model with LMI

The [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) is our recommended way of deploying LLMs using LMI on SageMaker.

Here is how you can deploy [TheBloke/Llama2-7b-fp16](https://huggingface.co/TheBloke/Llama-2-7B-fp16) model using LMI with the SDK:

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Setup role and sagemaker session
iam_role = sagemaker.get_execution_role() 
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

# Fetch the uri of the LMI DeepSpeed container that supports vLLM, LMI-Dist, and DeepSpeed backends
container_image_uri = image_uris.retrieve(framework="djl-deepspeed", version="0.26.0", region=region)

# Create the SageMaker Model object. In this example we'll use vLLM as our inference backend
model = Model(
  image_uri=container_image_uri,
  role=iam_role,
  env={
    "HF_MODEL_ID": "TheBloke/Llama-2-7B-fp16",
    "TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_ROLLING_BATCH": "vllm",
  }
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
endpoint_name = sagemaker.utils.name_from_base("llama-7b-endpoint")
model.deploy(instance_type="ml.g5.2xlarge", initial_instance_count=1, endpoint_name=endpoint_name)
predictor = Predictor(
  endpoint_name=endpoint_name,
  sagemaker_session=sagemaker_session,
  serializer=JSONSerializer(),
  deserializer=JSONDeserializer(),
)

# Make an inference request against the llama2-7b endpoint
outputs = predictor.predict({
  "inputs": "The diamondback terrapin was the first reptile to be",
  "parameters": {
    "do_sample": True,
    "max_new_tokens": 256,
  }
})
```

### Sample Notebooks
The following table provides notebooks that demonstrate how to deploy popular open source LLMs using LMI containers on SageMaker.
If this is your first time using LMI, or you want a starting point for deploying a specific model, we recommend following the notebooks below.

| Model                                                              | Instance Type      | Sample Notebook |
|--------------------------------------------------------------------|--------------------|-----------------|
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)      | `ml.g5.2xlarge`    | [notebook]()    |
| [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf)    | `ml.g5.12xlarge`   | [notebook]()    |
| [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf)    | `ml.p4d.24xlarge`  | [notebook]()    |
| [Llama-2-70b-AWQ](https://huggingface.co/TheBloke/Llama-2-70B-AWQ) | `ml.g5.12xlarge`   | [notebook]()    |
| [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)     | `ml.g5.2xlarge`    | [notebook]()    |
| [Mixtral-8x7b](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | `ml.p4d.24xlarge`  | [notebook]()    |
| [Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl)           | `ml.g5.12xlarge`   | [notebook]()    |
| [CodeLlama-34b](https://huggingface.co/codellama/CodeLlama-34b-hf) | `ml.g5.48xlarge`   | [notebook]()    |
| [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b)               | `ml.g5.2xlarge`    | [notebook]()    |
| [Falcon-40b](https://huggingface.co/tiiuae/falcon-40b)             | `ml.g5.48xlarge`   | [notebook]()    |
| [Falcon-180b](https://huggingface.co/tiiuae/falcon-180b)           | `ml.p4de.24xlarge` | [notebook]()    |            

**Note: Some models in the table above are available from multiple providers. 
We link to the specific model we tested with, but we expect same model from a different provider (or a fine-tuned variant) to work.**

### Deployment Guide

We have put together a comprehensive [deployment guide](deployment_guide/README.md) that takes you through the steps needed to deploy a model using LMI containers on SageMaker.
The document covers the phases from storing your model artifacts through benchmarking your SageMaker endpoint.

## Supported LMI Inference Libraries

LMI Containers provide integration with multiple inference libraries.
You can learn more about their integration with LMI from the respective user guides:

* [DeepSpeed - User Guide](user_guides/deepspeed_user_guide.md)
* [vLLM - User Guide](user_guides/vllm_user_guide.md)
* [LMI-Dist - User Guide](user_guides/lmi-dist_user_guide.md)
* [TensorRT-LLM - User Guide](user_guides/trt_llm_user_guide.md)
* [Transformers NeuronX - User Guide](user_guides/tnx_user_guide.md)

LMI provides access to multiple libraries to enable users to find the best stack for their model and use-case. 
Each inference framework provides a unique set of features and optimizations that can be tuned for your model and use case.
With LMIs built-in inference handlers and unified configuration, experimenting with different stacks is as simple as changing a few configurations.
Refer to the stack specific user guides, and the [LMI deployment guide](deployment_guide/README.md) to learn more.
An overview of the different LMI components is provided in the [deployment guide](deployment_guide/README.md#components-of-lmi)

The following table shows which SageMaker DLC (deep learning container) to use for each backend.
This information is also available on the SageMaker DLC [GitHub repository](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers).

| Backend                | SageMakerDLC    | Example URI                                                                              |
|------------------------|-----------------|------------------------------------------------------------------------------------------|
| `vLLM`                 | djl-deepspeed   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-deepspeed0.12.6-cu121  |
| `lmi-dist`             | djl-deepspeed   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-deepspeed0.12.6-cu121  |
| `deepspeed`            | djl-deepspeed   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-deepspeed0.12.6-cu121  |
| `tensorrt-llm`         | djl-tensorrtllm | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-tensorrtllm0.7.1-cu122 |
| `transformers-neuronx` | djl-neuronx     | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-neuronx-sdk2.16.0      |
