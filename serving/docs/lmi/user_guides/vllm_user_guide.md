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

We recommend that regardless of which quantization technique you are using that you pre-quantize the model.
Runtime quantization adds additional overhead to the endpoint startup time.
Depending on the quantization technique, this can be significant overhead.
If you are using a pre-quantized model, you should not set any quantization specific configurations.
vLLM will deduce the quantization from the model config, and apply optimizations at runtime.
If you explicitly set the quantization configuration for a pre-quantized model, it limits the optimizations that vLLM can apply.

### Runtime Quantization

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
Following that, you can configure the LoRA support through the additional settings:

  - `option.max_loras`
  - `option.max_lora_rank`
  - `option.lora_extra_vocab_size`
  - `option.max_cpu_loras`
  - 
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

The following table lists the set of LMI configurations.
In some situations, the equivalent vLLM configuration can be used interchangeably. 
Those situations will be called out specifically.

| Item                                    | LMI Version | vLLM alias                    | Example Value | Default Value                              | Description                                                                                                  |                                                                                               
|-----------------------------------------|-------------|-------------------------------|---------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| option.quantize                         | \>= 0.23.0  | option.quantization           | awq           | None                                       | The quantization algorithm to use. See "Quantization Support" for more details                               |
| option.max_rolling_batch_prefill_tokens | \>= 0.24.0  | option.max_num_batched_tokens | 32768         | None                                       | Maximum number of tokens that the engine can process in a single batch iteration (includes prefill + decode) |
| option.cpu_offload_gb_per_gpu           | \>= 0.29.0  | option.cpu_offload_gb | 4 (GB)        | 0                                          | The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. |

In addition to the configurations specified in the table above, LMI supports all additional vLLM EngineArguments in Pass-Through mode.
Pass-Through configurations are not processed or validated by LMI.
You can find the set of EngineArguments supported by vLLM [here](https://docs.vllm.ai/en/v0.6.3.post1/models/engine_args.html#engine-args).

You can specify these pass-through configurations in the serving.properties file by prefixing the configuration with `option.<config>`,
or as environment variables by prefixing the configuration with `OPTION_<CONFIG>`.

We will consider two examples: a boolean configuration, and a string configuration.

**Boolean Configuration**

If you want to set the Engine Argument `enable_prefix_caching`, you can do:

* `option.enable_prefix_caching=true` in serving.properties
* `OPTION_ENABLE_PREFIX_CACHING=true` as an environment variable

**String configuration**

If you want to set the Engine Argument `tokenizer_mode`, you can do:

* `option.tokenizer_mode=mistral` in serving.properties
* `OPTION_TOKENIZER_MODE=true` in serving.properties
