# LMI V10 containers release

This document will contain the latest releases of our LMI containers. For details on any other previous releases, please refer our [github release page](https://github.com/deepjavalibrary/djl-serving/releases)

## Release Notes

### Release date: June 6, 2024
Check out our latest [Large Model Inference Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers).

### Key Features
#### LMI container
* Provided general performance optimization.
* **Added text embedding support**
  * Our solution for text embedding is 5% faster than HF TEI solution.
* Multi-LoRA feature now supports LLama3 and AWS models

#### TensorRT-LLM container
* Upgraded to TensorRT-LLM 0.9.0
* AWQ, FP8 support for Llama3 models on G6/P5 machines
* Now, default max_new_tokens=16384
* Bugfix for critical memory leaks on long run. 
* Bugfix for model hanging issues.

#### Transformers NeuronX container
* Upgraded to Transformers NeuronX 2.18.2

#### DeepSpeed container (deprecated)
DeepSpeed container is now deprecated. If you are not using `deepspeed` engine, all you need is `0.28.0-lmi` container and continue using it. 

### CX Usability Enhancements/Changes
* Model loading CX:
    - SERVING_LOAD_MODELS env is deprecated, use HF_MODEL_ID instead.
* Inference CX:
    - Input/Output schema changes:
        - Speculative decoding now in streaming, returns multiple jsonlines tokens at each generation step
        - Standardized the output formatter signature:
            - We reduced the parameters of output_formatter by introducing RequestOutput class.
            - RequestOutput contains all input information such as text, token_ids and parameters and also output information such as output tokens, log probabilities, and other details like finish reason. Check this [doc](https://github.com/deepjavalibrary/djl-serving/blob/e4d7e5da822a8c11b13e79eaeaec4101fe678b69/serving/docs/lmi/user_guides/lmi_input_output_schema.md#generationparameters) to know more.
            - Introduced prompt details in the `details` of the response for vLLM and lmi-dist rolling batch options. These prompt details contains the prompt token_ids and their corresponding text and log probability. Check this [doc](https://github.com/deepjavalibrary/djl-serving/blob/e4d7e5da822a8c11b13e79eaeaec4101fe678b69/serving/docs/lmi/user_guides/output_formatter_schema.md#custom-output-formatter-schema) to know more.
    * New error handling mechanism:
        - Improved our error handling for container responses for rolling batch. Check this [doc](https://github.com/deepjavalibrary/djl-serving/blob/e4d7e5da822a8c11b13e79eaeaec4101fe678b69/serving/docs/lmi/user_guides/lmi_input_output_schema.md#error-responses) to know more
    * New CX capability:
        - We introduce OPTION_TGI_COMPAT env which enables you to get the same response format as TGI. [doc]()
        - We also now support SSE text/event-stream data format. 

### Breaking Changes
* Inference CX for rolling batch:
    - Token id changed from list into integer in rolling batch response.
    - Error handling: “finish_reason: error” during rolling batch inference
* DeepSpeed container has been deprecated, functionality is generally available in the LMI container now

### Known Issues
- LMI-TRTLLM container
  * TRTLLM periodically crashes during model compilation
  * TRTLLM AWQ quantization currently crashes due to an internal error