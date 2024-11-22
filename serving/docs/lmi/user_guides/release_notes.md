# LMI V12 DLC containers release

This document will contain the latest releases of our LMI containers for use on SageMaker. 
For details on any other previous releases, please refer our [github release page](https://github.com/deepjavalibrary/djl-serving/releases)

## Release Notes

### Key Features

#### DJL Serving Changes (applicable to all containers)
* Fixed a bug related to HTTP error code and response handling when using a rolling batch/continuous batching engine:
  * When the python process returned outputs back to the frontend, the frontend was not using the provided HTTP error code (always returned 200)
* For all inference backends, we now rely on the tokenizer created by the engine for all processing. Previously, there were some cases where we created a separate tokenizer for processing.
* Enabled specified Java logging level to apply to Python process log level
  * For example, if you set `SERVING_OPTS="-Dai.djl.logging.level=debug"`, this will also enable debug level logging on the python code
* Improved validation logic on request schema and improved returned validation exception messages
* Added requestId logging for better per-request visibility and debugging
* Fixed a race condition that could result in a model worker dying for seemingly no reason
  * If a request resulted in an error such that the python process was restarted, during the restart it was possible for a new request to trump
    the restart process. As a result, the frontend lost knowledge of the restart progress and would shut down the worker after `model_loading_timeout` seconds.


#### LMI Container (vllm, lmi-dist) - Release 10-28-2024
* vLLM updated to version 0.6.2 
* Added support for new multi-modal models including pixtral and Llama3.2
* Added support for Tensor Parallel + Pipeline Parallel execution to support multi-node inference
* Various performance improvements and enhacements for the lmi-dist engine
* Please make note of specific behavior changes documented in the [breaking changes](../announcements/breaking_changes.md) section.


#### TensorRT-LLM Container - Release 11-15-2024
* TensorRT-LLM updated to version 0.12.0
* Support for Llama3.1 models
* Please make note of specific behavior changes documented in the [breaking changes](../announcements/breaking_changes.md) section.


#### Transformers NeuronX Container - Release 11-20-2024
* Neuron artifacts are updated to 2.20.1 
* Transformers neuronx is updated to 0.12.313
* Vllm is updated to 0.6.2
* Compilation time improvement. HF model can directly be loaded into NeuronAutoModel, so split and save step is no longer needed.


#### Text Embedding (using the LMI container)
* Various performance improvements
