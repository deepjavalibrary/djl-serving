# Steps for Deploying models with LMI Containers on AWS SageMaker

The following document provides a step-by-step guide for deploying LLMs using LMI Containers on AWS SageMaker.
This is an in-depth guide that will cover all phases from model artifacts through benchmarking your endpoint.
If this is your first time using LMI, we highly recommend you start with one of our [example notebooks](../README.md#sample-notebooks) to get familiar with LMI and SageMaker.

Before starting this tutorial, you should have the HuggingFace ModelId (e.g. `TheBloke/Llama-2-13b-fp16`) of the model you aim to deploy.
If you have a custom model, it must be saved in the HuggingFace Transformers Pretrained format.
You can read [this guide](model-artifacts.md) to verify your model is saved in the correct format for LMI.

This guide is organized as follows:

- [1. Instance Type Selection](instance-type-selection.md)
  - Pick a SageMaker instance type based on your model size and expected runtime usage 
- [2. Backend Selection](backend-selection.md)
  - Pick a backend (vLLM, TensorRT-LLM, DeepSpeed, LMI-Dist, Transformers NeuronX) and corresponding container
- [3. Model and Container Configurations](configurations.md)
  - Configure your model for optimized performance based on your use-case 
- [4. Deploying a SageMaker Endpoint](deploying-your-endpoint.md)
  - Deploy your model to a SageMaker Endpoint and make inference requests
- [5. Benchmarking your setup](benchmarking-your-endpoint.md)
  - Benchmark your setup to determine whether additional tuning is needed
- [6. Advanced Guides]()
  - A collection of advanced guides to further tune your deployment

Next: [Selecting an Instance Type](instance-type-selection.md)

Below we provide an overview of the various components of LMI containers.
We recommend reading this overview to become familiar with some of the LMI specific terminology like Backends and Built-In Handlers.

## Components of LMI

LMI containers bundle together a model server, LLM inference libraries, and inference handler code to deliver a batteries included LLM serving solution.
The model server, inference libraries, and default inference handler code are brought together through a unified configuration that specifies your deployment setup.
Brief overviews of the components relevant to LMI are presented below.

### Model Server
LMI containers use [DJL Serving](https://github.com/deepjavalibrary/djl-serving) as the Model Server.
A full architectural overview of DJL Serving is available [here](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/architecture.md).
At a high level, DJL Serving consists of Netty front-end that manages request routing to the backend workers that execute inference for the target model.
The backend manages model worker scaling, with each model being executed in an Engine.
The Engine is an abstraction provided by DJL that you can think of as the interface that allows DJL to run inference for a model with a specific deep learning framework.
In LMI, we use the Python Engine as it allows us to directly leverage the growing Python ecosystem of LLM inference libraries.

### Python Engine and Inference Backends
The Python Engine allows LMI containers to leverage many Python based inference libraries like vLLM, DeepSpeed, TensorRT-LLM, and Transformers NeuronX.
These libraries expose Python APIs for loading and executing models with optimized inference on accelerators like GPUs and AWS Inferentia.
LMI containers integrate the front-end model server with backend workers running Python processes to provide high performance inference of LLMs.

To support multi-gpu inference of large models using model parallelism techniques like tensor parallelism, many of the inference libraries support distributed inference through MPI.
LMI supports running the Python Engine in mpi mode (referred to as the MPI Engine) to leverage tensor parallelism in mpi aware libraries like DeepSpeed, LMI-Dist, and TensorRT-LLM.

Throughout the LMI documentation, we will use the term `backend` to refer to a combination of Engine and Inference Library (e.g. MPI Engine + LMI-Dist library).

### Built-In Handlers
LMI provides built-in inference handlers for all the supported backend.
These handlers take care of parsing configurations, loading the model onto accelerators, applying optimizations, and executing inference.
These libraries expose features and capabilities through different APIs and mechanisms, so switching between frameworks to maximize performance can be difficult as you need to learn each framework individually.
With LMI's built-in handlers, there is no need to learn each library and write custom code to leverage the features and optimizations they offer.
We expose a unified configuration format that allows you to easily switch between libraries as they evolve and improve over time.
As the ecosystem grows and new libraries become available, LMI can integrate them and offer the same consistent experience.

### Configuration

The configuration provided to LMI specifies your entire setup. The configuration covers many aspects including:

* Where your model artifacts are stored (HuggingFace ModelId, S3 URI)
* Model Server Configurations like job/request queue size, auto-scaling behavior for model workers, which engine to use (either Python or MPI for LMI)
* Engine/Backend Configurations like whether to use quantization, input sequence limits, continuous batching size, tensor parallel degree, and more depending on the specific backend you use

Configurations can be provided as either a `serving.properties` file, or through environment variables passed to the container.
A more in-depth explanation about configurations is presented in the deployment guide in the [Container and Model Configurations](configurations.md) section.

## Feature Matrix

|                                       | HuggingFace Accelerate | LMI_dist (8.0.0) | DeepSpeed (0.12.6)                                                                                                       | TensorRTLLM (0.7.1)                                                                                                           | TransformersNeuronX (2.16.0)                                                                                                                  | vLLM (0.2.7) |
|---------------------------------------|------------------------| ---------------- |--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------| ------------ |
| DLC                                   | DeepSpeed              | DeepSpeed        | DeepSpeed                                                                                                                | LMI TRTLLM                                                                                                                    | LMI Neuron                                                                                                                                    | DeepSpeed    |
| Default handler                       | [huggingface](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/huggingface.py)        | [huggingface](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/huggingface.py)      | [deepspeed](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/deepspeed.py) | [tensorrtllm](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/tensorrt_llm.py) | [transformersneuronx](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/transformers_neuronx.py) | [huggingface](https://github.com/deepjavalibrary/djl-serving/blob/0.26.0-dlc/engines/python/setup/djl_python/huggingface.py)  |
| support quantization                  | BitsandBytes/GPTQ      | GPTQ/AWQ         | SmoothQuant INT8                                                                                                         | SmoothQuant, AWQ, GPTQ                                                                                                        | INT8                                                                                                                                          | GPTQ/AWQ     |
| AWS machine supported                 | G4/G5/P4D/P5           | G5/P4D/P5        | G5/P4D/P5                                                                                                                | G5/P4D/P5                                                                                                                     | INF2/TRN1                                                                                                                                     | G4/G5/P4D/P5 |
| execution mode                        | Python                 | MPI              | MPI                                                                                                                      | MPI                                                                                                                           | Python                                                                                                                                        | Python       |
| multi-accelerator weight loading      | Yes                    | Yes              | Yes                                                                                                                      | Yes                                                                                                                           | Yes                                                                                                                                           | Yes          |
| tensor parallel                       | No                     | Yes              | Yes                                                                                                                      | Yes                                                                                                                           | Yes                                                                                                                                           | Yes          |
| continuous batching streaming         | Yes                    | Yes              | Yes                                                                                                                      | Yes                                                                                                                           | Yes                                                                                                                                           | Yes          |
| need to compile                       | No                     | No               | No                                                                                                                       | Yes                                                                                                                           | Yes                                                                                                                                           | No           |
| SageMaker Inference Component support | Yes                    | Yes              | Yes                                                                                                                      | Yes                                                                                                                           | Yes                                                                                                                                           | Yes          |
| support logprob                       | Yes                    | Yes              | Yes                                                                                                                      | Yes                                                                                                                           | No                                                                                                                                            | Yes          |
