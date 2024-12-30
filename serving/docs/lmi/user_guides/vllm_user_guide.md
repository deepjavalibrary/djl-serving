# vLLM Engine User Guide

## Model Artifacts Structure

vLLM expects the model artifacts to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md#huggingface-transformers-pretrained-format).

## Supported Model architecture

**Text Generation Models**

Here is the list of text generation models supported in [vllm 0.6.3.post1](https://docs.vllm.ai/en/v0.6.3.post1/models/supported_models.html#decoder-only-language-models).

**Multi Modal Models**

Here is the list of multi-modal models supported in [vllm 0.6.3.post1](https://docs.vllm.ai/en/v0.6.3.post1/models/supported_models.html#decoder-only-language-models).

### Model Coverage in CI

The following set of models are tested in our nightly tests

- GPT NeoX 20b
- Mistral 7b
- Phi2
- Starcoder2 7b
- Gemma 2b
- Llama2 7b
- Qwen2 7b - fp8
- Llama3 8b
- Falcon 11b
- Llava-v1.6-mistral (multi modal)
- Phi3v (multi modal)
- Pixtral 12b (multi modal)
- Llama3.2 11b (multi modal)

## Quantization Support

The quantization techniques supported in vLLM 0.6.3.post1 are listed [here](https://docs.vllm.ai/en/v0.6.3.post1/quantization/supported_hardware.html).

We highly recommend that regardless of which quantization technique you are using that you pre-quantize the model.
Runtime quantization adds additional overhead to the endpoint startup time, and depending on the quantization technique, this can be significant overhead.

The following quantization techniques are supported for runtime quantization:

- fp8
- bitsandbytes

You can leverage these techniques by specifying `option.quantize=<fp8|bitsandbytes>` in serving.properties, or `OPTION_QUANTIZE=<fp8|bitsandbytes>` environment variable.

Other quantization techniques supported by vLLM require ahead of time quantization to be served with LMI.
You can find details on how to leverage those quantization techniques from the vLLM docs [here](https://docs.vllm.ai/en/v0.6.3.post1/quantization/supported_hardware.html).

## Quick Start Configurations 

You can leverage `vllm` with LMI using the following starter configurations:

### serving.properties

```
engine=Python
option.tensor_parallel_degree=max
option.model_id=<your model>
option.rolling_batch=vllm
# Adjust the following based on model size and instance type
option.max_rolling_batch_size=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---servingproperties) to deploy a model with serving.properties configuration on SageMaker.

### environment variables 

```
HF_MODEL_ID=<your model>
TENSOR_PARALLEL_DEGREE=max
OPTION_ROLLING_BATCH=vllm
# Adjust the following based on model size and instance type
OPTION_MAX_ROLLING_BATCH_SIZE=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---environment-variables) to deploy a model with environment variable configuration on SageMaker.

### LoRA Adapter Support

vLLM has support for LoRA adapters using the [adapters API](../../adapters.md).
In order to use the adapters, you must begin by enabling them by setting `option.enable_lora=true`.
Following that, you can configure the LoRA support through the additional settings `option.max_loras`, `option.max_lora_rank`, `option.lora_extra_vocab_size`, and `option.max_cpu_loras`.
If you run into OOM by enabling adapter support, reduce the `option.gpu_memory_utilization`.

### Advanced vLLM Configurations

The following table lists the advanced configurations that are available with the vLLM backend.
There are two types of advanced configurations: `LMI`, and `Pass Through`.
`LMI` configurations are processed by LMI and translated into configurations that vLLM uses.
`Pass Through` configurations are passed directly to the backend library. These are opaque configurations from the perspective of the model server and LMI.
We recommend that you file an [issue](https://github.com/deepjavalibrary/djl-serving/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=) for any issues you encounter with configurations.
For `LMI` configurations, if we determine an issue with the configuration, we will attempt to provide a workaround for the current released version, and attempt to fix the issue for the next release.
For `Pass Through` configurations it is possible that our investigation reveals an issue with the backend library.
In that situation, there is nothing LMI can do until the issue is fixed in the backend library.

| Item                                    | LMI Version | Configuration Type | Description                                                                                                                                                                                                                                                                                                                                                                  | Example value         |
|-----------------------------------------|-------------|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| option.quantize                         | \>= 0.26.0  | LMI                | Quantize the model with the supported quantization methods. LMI uses this to set the right quantization configs in VLLM                                                                                                                                                                                                                                                      | `awq` Default: `None` |
| option.max_rolling_batch_prefill_tokens | \>= 0.26.0  | LMI                | Limits the number of tokens for prefill(a.k.a prompt processing). This needs to be tuned based on GPU memory available and request lengths. Setting this value too high can limit the number of kv cache blocks or run into OOM. If you don't set this, `vllm` will default to max model length from Hugging Face config(also accounts for rope scaling if applicable).      | Default: `None`       |
| option.max_model_len                    | \>= 0.26.0  | Pass Through       | the maximum length (input+output) vLLM should preserve memory for. If not specified, will use the default length the model is capable in config.json. Sometimes model's maximum length could go to 32k (Mistral 7B) and way beyond the supported KV token size. In that case to deploy on a small instance, we need to adjust this value within the range of KV Cache limit. | Default: `None`       |
| option.load_format                      | \>= 0.26.0  | Pass Through       | The checkpoint format of the model. Default is auto and means bin/safetensors will be used if found.                                                                                                                                                                                                                                                                         | Default: `auto`       |
| option.enforce_eager                    | \>= 0.27.0  | Pass Through       | vLLM by default will run with CUDA graph optimization to reach to the best performance. However, in the situation of very less GPU memory, having CUDA graph enabled will cause OOM. So if you set this option to true, we will use PyTorch Eager mode and disable CUDA graph to save some GBs of memory.                                                                    | Default: `False`      |
| option.gpu_memory_utilization           | \>= 0.27.0  | Pass Through       | This config controls the amount of GPU memory allocated to KV cache. Setting higher value will allocate more memory for KV cache.Default is 0.9. It recommended to reduce this value if GPU OOM's are encountered.                                                                                                                                                           | Default: `0.9`        |
| option.enable_lora                      | \>= 0.27.0  | Pass Through       | This config enables support for LoRA adapters.                                                                                                                                                                                                                                                                                                                               | Default: `false`      |
| option.max_loras                        | \>= 0.27.0  | Pass Through       | This config determines the maximum number of LoRA adapters that can be run at once. Allocates GPU memory for those number of adapters.                                                                                                                                                                                                                                       | Default: `4`          |
| option.max_lora_rank                    | \>= 0.27.0  | Pass Through       | This config determines the maximum rank allowed for a LoRA adapter. Set this value to maximum rank of your adapters. Setting a larger value will enable more adapters at a greater memory usage cost.                                                                                                                                                                        | Default: `16`         |
| option.lora_extra_vocab_size            | \>= 0.27.0  | Pass Through       | This config determines the maximum additional vocabulary that can be added through a LoRA adapter.                                                                                                                                                                                                                                                                           | Default: `256`        |
| option.max_cpu_loras                    | \>= 0.27.0  | Pass Through       | This config determines the maximum number of LoRA adapters to cache in memory. All others will be evicted to disk.                                                                                                                                                                                                                                                           | Default: `None`       |
| option.enable_chunked_prefill           | \>= 0.29.0  | Pass Through       | This config enables chunked prefill support. With chunked prefill, longer prompts will be chunked and batched with decode requests to reduce inter token latency. This option is EXPERIMENTAL and tested for llama and falcon models only. This does not work with LoRA and speculative decoding yet.                                                                        | Default: `None`       |
| option.cpu_offload_gb_per_gpu           | \>= 0.29.0  | Pass Through       | This config allows offloading model weights into CPU to enable large model running with limited GPU memory.                                                                                                                                                                                                                                                                  | Default: `0`          |
| option.enable_prefix_caching            | \>= 0.29.0  | Pass Through       | This config allows the engine to cache the context memory and reuse to speed up inference.                                                                                                                                                                                                                                                                                   | Default: `False`      |
| option.disable_sliding_window           | \>= 0.30.0  | Pass Through       | This config disables sliding window, capping to sliding window size inference.                                                                                                                                                                                                                                                                                               | Default: `False`      |
| option.tokenizer_mode                   | \>= 0.30.0  | Pass Through       | This config sets the tokenizer mode for vllm. When using mistral models with mistral tokenizers, you must set this to `mistral` explicitly.                                                                                                                                                                                                                                  | Default: `auto`       |

