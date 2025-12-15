# Release Notes

Below are the release notes for recent Large Model Inference (LMI) images for use on SageMaker.
For details on historical releases, refer to the [Github Releases page](https://github.com/deepjavalibrary/djl-serving/releases).

## LMI V18 (DJL-Serving 0.36.0)

Meet your brand new image! 💿

#### LMI (vLLM) Image – 12-15-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi18.0.0-cu128
```
* vLLM has been upgraded to `0.12.0`
* LMCache support for on-host caching of KV cache delivers up to 20x improvement to request latency for long context requests. Refer to [LMCache Performance Benefits for LMI Customers](../lmcache_performance.md) for more details
* Added support for adapter-scoped custom code (e.g., model.py) that can be registered dynamically via the adapter management APIs, enabling per-adapter input/output formatting for multi-tenant LoRA deployments

##### Key Features

**Enhanced Adapter Management with Custom Code Support**
* On adapter registration, DJL Serving now checks the adapter directory for `model.py` and (if present) loads the adapter's custom formatters before registering adapter weights
* If adapter custom code loading fails, registration fails fast (adapter weights are not registered) and returns an error response (code 424)
* During inference, adapter-specific formatters override base model formatters when the inference targets an adapter
* On adapter unregistration, the adapter's custom code is unloaded/cleaned up
* Enables per-adapter input/output formatting for multi-tenant LoRA deployments

**LMCache Performance Improvements**
* Up to 28x speedup in Time to First Token (TTFT) with CPU offloading (achieved with Qwen 2.5-7B at 2M token context length)
* Up to 16x speedup in TTFT with NVMe-based offloading (achieved with Qwen 2.5-72B at 1M token context length using O_DIRECT)
* Comprehensive benchmarking suite across different storage backends (CPU RAM, NVMe, Redis, S3, EBS)

**Security & Stability**
* Enhanced security validation for adapters in Secure Mode plugin
* Improved multimodal integration test stability with vLLM 0.12.0
* Updated CI/CD pipeline to use serving version consistently across workflows

## LMI V17 (DJL-Serving 0.35.0)

#### LMI (vLLM) Image – 9-30-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128
```
* vLLM has been upgraded to `0.11.1`
* Going forward, [async mode](https://github.com/deepjavalibrary/djl-serving/blob/0.35.0-dlc/serving/docs/lmi/user_guides/vllm_user_guide.md#async-mode-configurations) is the default configuration for the vLLM handler
* New models supported - DeepSeek V3.2, Qwen 3 VL, Minimax-M2
* LoRA supported in Async mode for MoE models -  Llama4 Scout, Qwen3, DeepSeek, GPT-OSS
* EAGLE 3 support added for GPT-OSS Models
* Support for on-host KV Cache offloading with [LMCache](https://github.com/deepjavalibrary/djl-serving/blob/0.35.0-dlc/serving/docs/lmi/user_guides/lmcache_user_guide.md) (LMCache v1 is in experimental phase).

##### Considerations
* Our benchmarks demonstrate improvement in performance of LMI V17 compared to V16 for all models benchmarked (DeepSeek R1 Distill Llama, Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3) except for Qwen3 Coder 30B A3b Model at concurrency of 128. We are working with vLLM community to understand the root cause and potential fixes.

## LMI V16 (DJL-Serving 0.34.0)

#### LMI (vLLM) Image – 9-30-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128
```
* vLLM version upgraded to `0.10.2`
* Going forward, [async mode](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/vllm_user_guide.md#async-mode-configurations) is the officially recommended configuration for the vLLM handler 
* Async vLLM handler now supports custom [input](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/input_formatter_schema.md) and [output](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/output_formatter_schema.md) formatters 
* Async vLLM handler now supports [multi-adapter](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/adapters.md) (LoRA) serving
* Async vLLM handler now supports session-based [sticky routing](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/stateful_sessions.md)

## LMI V15 (DJL-Serving 0.33.0)

#### LMI (vLLM) Image – 4-17-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128
```
* vLLM version upgraded to `0.8.4`
* Llama4 Model Support
* Updated Async Implementation, please see the [vLLM async user guide here](user_guides/vllm_user_guide.md#async-mode-configurations) 

#### TensorRT-LLM Image – 6-24-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128
```
* TensorRT-LLM version upgraded to `0.21.0rc1`
