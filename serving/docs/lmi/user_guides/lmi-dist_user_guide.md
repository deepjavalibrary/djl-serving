# LMI-Dist Engine User Guide

## Model Artifact Structure

LMI-Dist expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

**Text Generation Models**

LMI-Dist supports the same set of text-generation models as [vllm 0.6.3.post1](https://docs.vllm.ai/en/v0.6.3.post1/models/supported_models.html#decoder-only-language-models).

In addition to the vllm models, LMI-Dist also supports the t5 model family (e.g. google/flan-t5-xl).

**Multi Modal Models**

LMI-Dist supports the same set of multi-modal models as [vllm 0.6.3.post1](https://docs.vllm.ai/en/v0.6.3.post1/models/supported_models.html#decoder-only-language-models).

However, the one known exception is MLlama (Llama3.2 multimodal models). 
MLlama support is expected in the v13 (0.32.0) release.

### Model Coverage in CI

The following set of models are tested in our nightly tests:

- GPT NeoX 20b
- Falcon 7b
- Falcon2 11b
- GPT2
- MPT 7b
- Llama2 13b
- Llama3 8b
- Flan T5 Xl
- Octocoder
- Starcoder2 7b
- Gemma 2b
- Llama2 13b + GPTQ
- Mistral 7b
- Llama3.1 8b
- Llama3.1 8b - Multi Node
- Codestral 22b
- Llava-v1.6-mistral (multi modal)
- Phi3v (multi modal)
- Pixtral12b (multi modal)
- Llama3.2 1b/3b
- Llama2 13b + Speculative Decoding
- Codellama 34b
- Mixtral 8x7b
- DBRX
- Llama3.1 70b
- Baichuan2 13b
- Qwen1.5 14b

## Quick Start Configurations

You can leverage `lmi-dist` with LMI using the following starter configurations:

### serving.properties

```
engine=Python
option.mpi_mode=True
option.tensor_parallel_degree=max
option.rolling_batch=lmi-dist
option.model_id=<your model id>
# Adjust the following based on model size and instance type
option.max_rolling_batch_size=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-1-configuration---servingproperties) to deploy a model with serving.properties configuration on SageMaker.

### environment variables

```
HF_MODEL_ID=<your model id>
TENSOR_PARALLEL_DEGREE=max
OPTION_ROLLING_BATCH=lmi-dist
# Adjust the following based on model size and instance type
OPTION_MAX_ROLLING_BATCH_SIZE=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-2-configuration---environment-variables) to deploy a model with environment variable configuration on SageMaker.

### LoRA Adapter Support

LMI-Dist has support for LoRA adapters using the [adapters API](../../adapters.md).
In order to use the adapters, you must begin by enabling them by setting `option.enable_lora=true`.
Following that, you can configure the LoRA support through the additional settings `option.max_loras`, `option.max_lora_rank`, `option.lora_extra_vocab_size`, and `option.max_cpu_loras`.
If you run into OOM by enabling adapter support, reduce the `option.gpu_memory_utilization`.

Please check that your base model [supports LoRA adapters in vLLM](https://docs.vllm.ai/en/v0.5.3.post1/models/supported_models.html#decoder-only-language-models).

## Quantization Support

LMI-Dist supports the same quantization techniques as [vllm 0.6.3.post1](https://docs.vllm.ai/en/v0.6.3.post1/quantization/supported_hardware.html).

We highly recommend that regardless of which quantization technique you are using that you pre-quantize the model.
Runtime quantization adds additional overhead to the endpoint startup time, and depending on the quantization technique, this can be significant overhead.

The following quantization techniques are supported for runtime quantization:

- fp8
- bitsandbytes

You can leverage these techniques by specifying `option.quantize=<fp8|bitsandbytes>` in serving.properties, or `OPTION_QUANTIZE=<fp8|bitsandbytes>` environment variable.

Other quantization techniques supported by vLLM require ahead of time quantization to be served with LMI.
You can find details on how to leverage those quantization techniques from the vLLM docs [here](https://docs.vllm.ai/en/v0.6.2/quantization/supported_hardware.html).

### Ahead of time (AOT) quantization

If you bring a pre-quantized model to LMI, you should not set the `option.quantize` configuration.
The lmi-dist engine will directly parse the quantization configuration from the model and load it for inference.
This is especially important if you are using a technique that has a Marlin variant like GPTQ, AWQ, or FP8.
The engine will determine if it can use the Marlin kernels at runtime, and use them if it can (hardware support).

For example, let's say you are deploying an AWQ quantized model on a g6.12xlarge instance (GPUs support marlin).
If you explicitly specify `option.quantize=awq`, the engine will not apply Marlin as it is explicitly instructed to only use `awq`.
If you omit the `option.quantize` configuration, then the engine will determine it can use marlin and leverage that for optimized performance.
 

## Advanced LMI-Dist Configurations

Here are the advanced parameters that are available when using LMI-Dist.


| Item                                    | LMI Version | Configuration Type | Description                                                                                                                                                                                                                                                                                                                                                                                               | Example value         |
|-----------------------------------------|-------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| option.quantize                         | \>= 0.23.0  | LMI                | Quantize the model with the supported quantization methods(`gptq`, `awq`, `squeezellm`)                                                                                                                                                                                                                                                                                                                   | `awq` Default: `None` |
| option.max_rolling_batch_prefill_tokens | \>= 0.24.0  | Pass Through       | Limits the number of tokens for prefill(a.k.a prompt processing). This needs to be tuned based on GPU memory available and request lengths. Setting this value too high can limit the number of kv cache blocks or run into GPU OOM. If you don't set this, `lmi-dist` will default to max model length from Hugging Face config(also accounts for rope scaling if applicable).                           | Default: `None`       |
| option.max_model_len                    | \>= 0.27.0  | Pass Through       | The maximum length (input+output) of the request. The request will be stopped if more tokens are generated. `lmi-dist` will default to max model length from Hugging Face config(also accounts for rope scaling if applicable). For models with larger maximum length support(for e.g. 32k for Mistral 7B), it could lead to GPU OOM. In such cases, to deploy on a smaller instances, reduce this value. | Default: `None`       |
| option.load_format                      | \>= 0.27.0  | Pass Through       | The checkpoint format of the model. Default is auto and means bin/safetensors will be used if found.                                                                                                                                                                                                                                                                                                      | Default: `auto`       |
| option.enforce_eager                    | \>= 0.27.0  | Pass Through       | `lmi-dist` by default will run with CUDA graph optimization to reach to the best performance. However, in the situation of very less GPU memory, having CUDA graph enabled will cause OOM. So if you set this option to true, we will use PyTorch Eager mode and disable CUDA graph to save some GBs of memory. `T5` model will not use cuda graphs.                                                      | Default: `False`      |
| option.gpu_memory_utilization           | \>= 0.27.0  | Pass Through       | This config controls the amount of GPU memory allocated to KV cache. Setting higher value will allocate more memory for KV cache. Default is 0.9. It recommended to reduce this value if GPU OOM's are encountered.                                                                                                                                                                                       | Default: `0.9`        |
| option.speculative_draft_model          | \>= 0.27.0  | Pass Through       | Model id or path to speculative decoding draft model                                                                                                                                                                                                                                                                                                                                                      | Default: `None`       |
| option.draft_model_tp_size              | \>= 0.27.0  | Pass Through       | Tensor parallel degree of speculative decoding draft model. Accepted values are `1` and target model's tensor parallel size(`option.tensor_parallel_degree`)                                                                                                                                                                                                                                              | Default: `1`          |
| option.speculative_length               | \>= 0.27.0  | Pass Through       | Determines the number of tokens draft model generates before verifying against target model                                                                                                                                                                                                                                                                                                               | Default: `5`          |
| option.record_acceptance_rate           | \>= 0.27.0  | LMI                | Enables logging speculative decoding acceptance rate                                                                                                                                                                                                                                                                                                                                                      | Default: `False`      |
| option.enable_lora                      | \>= 0.27.0  | Pass Through       | This config enables support for LoRA adapters.                                                                                                                                                                                                                                                                                                                                                            | Default: `false`      |
| option.max_loras                        | \>= 0.27.0  | Pass Through       | This config determines the maximum number of LoRA adapters that can be run at once. Allocates GPU memory for those number adapters.                                                                                                                                                                                                                                                                       | Default: `4`          |
| option.max_lora_rank                    | \>= 0.27.0  | Pass Through       | This config determines the maximum rank allowed for a LoRA adapter. Set this value to maximum rank of your adapters. Setting a larger value will enable more adapters at a greater memory usage cost.                                                                                                                                                                                                     | Default: `16`         |
| option.lora_extra_vocab_size            | \>= 0.27.0  | Pass Through       | This config determines the maximum additional vocabulary that can be added through a LoRA adapter.                                                                                                                                                                                                                                                                                                        | Default: `256`        |
| option.max_cpu_loras                    | \>= 0.27.0  | Pass Through       | This config determines the maximum number of LoRA adapters to cache in memory. All others will be evicted to disk.                                                                                                                                                                                                                                                                                        | Default: `None`       |
| option.enable_chunked_prefill           | \>= 0.29.0  | Pass Through       | This config enables chunked prefill support. With chunked prefill, longer prompts will be chunked and batched with decode requests to reduce inter token latency. This option is EXPERIMENTAL and tested for llama and falcon models only. This does not work with LoRA and speculative decoding yet.                                                                                                     | Default: `None`       |
| option.cpu_offload_gb_per_gpu           | \>= 0.29.0  | Pass Through       | This config allows offloading model weights into CPU to enable large model running with limited GPU memory.                                                                                                                                                                                                                                                                                               | Default: `0`          |
| option.enable_prefix_caching            | \>= 0.29.0  | Pass Through       | This config allows the engine to cache the context memory and reuse to speed up inference.                                                                                                                                                                                                                                                                                                                | Default: `False`      |
| option.disable_sliding_window           | \>= 0.30.0  | Pass Through       | This config disables sliding window, capping to sliding window size inference.                                                                                                                                                                                                                                                                                                                            | Default: `False`      |
| option.tokenizer_mode                   | \>= 0.30.0  | Pass Through       | This config sets the tokenizer mode for vllm. When using mistral models with mistral tokenizers, you must set this to `mistral` explicitly.                                                                                                                                                                                                                                                               | Default: `auto`       |
