# LMI V15 DLC containers release

This document will contain the latest releases of our LMI containers for use on SageMaker. 
For details on any other previous releases, please refer our [github release page](https://github.com/deepjavalibrary/djl-serving/releases)

## Release Notes

### Key Features

#### LMI Container (vllm) - Release 4-17-2025
* vLLM updated to version 0.8.4
* Llama4 Model Support
* Updated Async Implementation, please see the [vLLM async user guide here](user_guides/vllm_user_guide.md#async-mode-configurations). 

#### TensorRT-LLM Container - Coming Soon 
We plan to update our TensorRT-LLM integration in LMI v15.
This update will include

* Integration with TensorRT-LLM version 0.18.2
* Deprecation of Rolling Batch support, and replacement with Async Engine support