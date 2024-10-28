# LMI-Dist Engine User Guide

## Model Artifact Structure

LMI-Dist expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

The model architecture that we test for lmi-dist (in CI):

- DBRX
- Falcon
- Gemma
- GPT-NeoX
- Llama3.1 (un-quantized and with GPTQ)
- Mistral (un-quantized and with AWQ)
- Mixtral
- MPT
- Octocoder
- Phi
- Starcoder
- T5
- LlaVA-NeXT
- Phi-3-Vision
- Pixtral


### Complete Model Set

LMI Dist supports the same set of models as vLLM. For LMI v12, we use vLLM 0.6.2. The set of supported models
for vLLM 0.6.2 can be found [here](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html).

**Note: MLlama (Llama3.2 multimodal models) are currently not supported in the LMI-Dist backend. 
You should use the vLLM backend for those models.**

In addition to vLLM's supported models, LMI-Dist also support t5 model architectures like [`flan-t5-xl`](https://huggingface.co/google/flan-t5-xl).

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

vLLM has support for LoRA adapters using the [adapters API](../../adapters.md).
In order to use the adapters, you must begin by enabling them by setting `option.enable_lora=true`.
Following that, you can configure the LoRA support through the additional settings `option.max_loras`, `option.max_lora_rank`, `option.lora_extra_vocab_size`, and `option.max_cpu_loras`.
If you run into OOM by enabling adapter support, reduce the `option.gpu_memory_utilization`.

Please check that your base model [supports LoRA adapters in vLLM](https://docs.vllm.ai/en/v0.5.3.post1/models/supported_models.html#decoder-only-language-models).


## Quantization Support

Currently, we allow customer to use `option.quantize=<quantization-type>` or `OPTION_QUANTIZE=<quantization-type>` to load a quantized model in `lmi-dist`.

We support the following `<quantization-type>`:

* awq (LMI container versions >= 0.26.0)
* gptq (LMI container versions >= 0.24.0)
* squeezellm (LMI container versions >= 0.27.0)

When using pre-quantized models make sure to use the correct model artifacts e.g. `TheBloke/Llama-2-13B-chat-GPTQ`, `TheBloke/Llama-2-13B-chat-AWQ`.

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
