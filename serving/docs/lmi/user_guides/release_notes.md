# LMI V10 containers release

This document will contain the latest releases of our LMI containers. For details on any other previous releases, please refer our [github release page](https://github.com/deepjavalibrary/djl-serving/releases)

## Release Notes

### Release date: August 16, 2024

Check out our latest [Large Model Inference Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers).

### Key Features

#### DJL Serving Changes (applicable to all containers)
* Allows configuring health checks to fail based on various types of error rates
* When not streaming responses, all invocation errors will respond with the appropriate 4xx or 5xx HTTP response code
  * Previously, for some inference backends (vllm, lmi-dist, tensorrt-llm) the behavior was to return 2xx HTTP responses when errors occurred during inference
* HTTP Response Codes are now configurable if you require a specific 4xx or 5xx status to be returned in certain situations
* Introduced annotations `@input_formatter` and `@output_formatter` to bring your own script for pre- and post-postprocessing.


#### LMI Container (vllm, lmi-dist)
* vLLM updated to version 0.5.3.post1
* Added MultiModal Support for Vision Language Models using the OpenAI Chat Completions Schema.
  * More details available [here](https://github.com/deepjavalibrary/djl-serving/blob/v0.29.0/serving/docs/lmi/user_guides/vision_language_models.md)
* Supports Llama 3.1 models
* Supports beam search, `best_of` and `n` with non streaming output. 
* Supports chunked prefill support in both vllm and lmi-dist.


#### TensorRT-LLM Container
* TensorRT-LLM updated to version 0.11.0
* **[Breaking change]** Flan-T5 is now supported with C++ triton backend. Removed Flan-T5 support for TRTLLM python backend.


#### Transformers NeuronX Container
* Upgraded to Transformers NeuronX 2.19.1


#### Text Embedding (using the LMI container)
* Various performance improvements


### Breaking Changes
* In the TensorRT-LLM container, Flan-T5 is now supported with C++ triton backend. Removed Flan-T5 support for TRTLLM python backend.

### Known Issues
* Running Gemma and Phi models with TensorRT-LLM is only viable currently at TP=1 because of an issue in TensorRT-LLM where one engine is built even when TP > 1.