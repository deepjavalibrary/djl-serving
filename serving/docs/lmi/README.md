# Table of Contents

- [Overview - Large Model Inference (LMI) Containers](#overview---large-model-inference-lmi-containers)
- [QuickStart](#quickstart)
  - [Sample Notebooks](#sample-notebooks)
  - [Starting Guide](#starting-guide)
  - [Advanced Deployment Guide](#advanced-deployment-guide)
- [Supported LMI Inference Libraries](#supported-lmi-inference-libraries)

# Overview - Large Model Inference (LMI) Containers

LMI containers are a set of high-performance Docker Containers purpose built for large language model (LLM) inference. 
With these containers, you can leverage high performance open-source inference libraries like [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), 
[Transformers NeuronX](https://github.com/aws-neuron/transformers-neuronx) to deploy LLMs on [AWS SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html). 
These containers bundle together a model server with open-source inference libraries to deliver an all-in-one LLM serving solution.
We provide quick start notebooks that get you deploying popular open source models in minutes, and advanced guides to maximize performance of your endpoint.

LMI containers provide many features, including:

* Optimized inference performance for popular model architectures like Llama, Bloom, Falcon, T5, Mixtral, and more
* Integration with open source inference libraries like vLLM, TensorRT-LLM, and Transformers NeuronX
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

Our recommended progression for the LMI documentation is as follows:

1. [Sample Notebooks](#sample-notebooks): We provide sample notebooks for popular models that can be run as-is. This is the quickest way to start deploying models with LMI.
2. [Starting Guide](#starting-guide): The starter guide describes a simplified UX for configuring LMI containers. This UX is applicable across all LMI containers, and focuses on the most important configurations available for tuning performance.
3. [Deployment Guide](#advanced-deployment-guide): The deployment guide is an advanced guide tailored for users that want to squeeze the most performance out of LMI. It is intended for users aiming to deploy LLMs in a production setting, using a specific backend.

### Sample Notebooks
The following table provides notebooks that demonstrate how to deploy popular open source LLMs using LMI containers on SageMaker.
If this is your first time using LMI, or you want a starting point for deploying a specific model, we recommend following the notebooks below.
All the below samples are hosted in the [SageMaker Generative AI Hosting Examples Repository](https://github.com/aws-samples/sagemaker-genai-hosting-examples).
That repository will be continuously updated with examples.

| Model                                                              | Instance Type     | Sample Notebook                                                                                                                                 |
|--------------------------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)      | `ml.g5.2xlarge`   | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-7b/LMI/llama2-7b.ipynb)                      |
| [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf)    | `ml.g5.12xlarge`  | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-13b/LMI/llama2-13b.ipynb)                    |
| [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf)    | `ml.p4d.24xlarge` | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-70b/LMI/llama2-70b.ipynb)                    |
| [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)     | `ml.g5.2xlarge`   | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Mistral/Mistral-7b/LMI/mistral-lmi-sme-dept.ipynb)         |
| [Mixtral-8x7b](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | `ml.p4d.24xlarge` | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Mixtral/Mixtral-8x7b/LMI/mixtral-8x7b-trtllm-deploy.ipynb) |
| [Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl)           | `ml.g5.12xlarge`  | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/FlanT5/LMI/flant5-xxl.ipynb)                               |
| [CodeLlama-13b](https://huggingface.co/codellama/CodeLlama-34b-hf) | `ml.g5.48xlarge`  | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/CodeLlama/CodeLlama-13b/LMI/codellama-13b.ipynb)           |
| [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b)               | `ml.g5.2xlarge`   | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Falcon/Falcon-7B/LMI/falcon-7b-trt-llm.ipynb)              |
| [Falcon-40b](https://huggingface.co/tiiuae/falcon-40b)             | `ml.g5.48xlarge`  | [notebook](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Falcon/Falcon-40B/LMI/falcon-40b-trt-llm.ipynb)            |

**Note: Some models in the table above are available from multiple providers. 
We link to the specific model we tested with, but we expect same model from a different provider (or a fine-tuned variant) to work.**

### Starting Guide

The [starting guide](user_guides/starting-guide.md) is our recommended introduction for all users. 
This guide provides a simplified UX through a reduced set of configurations that are applicable to all LMI containers starting from v0.27.0. 

### Advanced Deployment Guide

We have put together a comprehensive [deployment guide](deployment_guide/README.md) that takes you through the steps needed to deploy a model using LMI containers on SageMaker.
The document covers the phases from storing your model artifacts through benchmarking your SageMaker endpoint.
It is intended for users moving towards deploying LLMs in production settings.

## Supported LMI Inference Libraries

LMI Containers provide integration with multiple inference libraries.
You can learn more about their integration with LMI from the respective user guides:

* [vLLM - User Guide](user_guides/vllm_user_guide.md)
* [LMI-Dist - User Guide](user_guides/lmi-dist_user_guide.md)
* [TensorRT-LLM - User Guide](user_guides/trt_llm_user_guide.md)
* [Transformers NeuronX - User Guide](user_guides/tnx_user_guide.md)
* [HuggingFace Accelerate - User Guide](user_guides/hf_accelerate.md)

LMI provides access to multiple libraries to enable users to find the best stack for their model and use-case. 
Each inference framework provides a unique set of features and optimizations that can be tuned for your model and use case.
With LMIs built-in inference handlers and unified configuration, experimenting with different stacks is as simple as changing a few configurations.
Refer to the stack specific user guides, and the [LMI deployment guide](deployment_guide/README.md) to learn more.
An overview of the different LMI components is provided in the [deployment guide](deployment_guide/README.md#components-of-lmi)

The following table shows which SageMaker DLC (deep learning container) to use for each backend.
This information is also available on the SageMaker DLC [GitHub repository](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers).

| Backend                | SageMakerDLC    | Example URI                                                                              |
|------------------------|-----------------|------------------------------------------------------------------------------------------|
| `vLLM`                 | djl-lmi         | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124        |
| `lmi-dist`             | djl-lmi         | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124        |
| `hf-accelerate`        | djl-lmi         | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124        |
| `tensorrt-llm`         | djl-tensorrtllm | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0-tensorrtllm0.9.0-cu122 |
| `transformers-neuronx` | djl-neuronx     | 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2      |

## Advanced Features

LMI contains also contain several advanced features that can be used for more complicated behaviors:

- [Workflow Support](../workflows.md)
- [Adapters Support](../adapters.md) (LoRA)
