# LMI-Dist Engine User Guide

## Model Artifact Structure

LMI-Dist expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

The model architecture that we test for lmi-dist (in CI):

- Llama (un-quantized and with GPTQ)
- Falcon
- GPT-NeoX
- MPT
- Mistral (un-quantized and with AWQ)
- Mixtral
- T5
- gemma
- starcoder
- phi
- dbrx


### Complete Model Set

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct`, etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen2-beta-7B`, `Qwen/Qwen2-beta-7B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)
- T5 (`google/flan-t5-xxl`, `google/flan-t5-base`, etc.)

We will add more model support for the future versions to have them tested. Please feel free to [file us an issue](https://github.com/deepjavalibrary/djl-serving/issues/new/choose) for more model coverage in CI.


## Quick Start Configurations

You can leverage `lmi-dist` with LMI using the following starter configurations:

### serving.properties

```
engine=MPI
option.tensor_parallel_degree=max
option.rolling_batch=lmi-dist
option.model_id=<your model id>
# Adjust the following based on model size and instance type
option.max_rolling_batch_size=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---servingproperties) to deploy a model with serving.properties configuration on SageMaker.

### environment variables

```
HF_MODEL_ID=<your model id>
TENSOR_PARALLEL_DEGREE=max
OPTION_ROLLING_BATCH=lmi-dist
# Adjust the following based on model size and instance type
OPTION_MAX_ROLLING_BATCH_SIZE=64
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---environment-variables) to deploy a model with environment variable configuration on SageMaker.

## Quantization Support

Currently, we allow customer to use `option.quantize=<quantization-type>` or `OPTION_QUANTIZE=<quantization-type>` to load a quantized model in `lmi-dist`.

We support the following `<quantization-type>`:

* awq (LMI container versions >= 0.26.0)
* [DEPRECATED] bitsandbytes (LMI container versions >= 0.24.0 and < 0.27.0)
* [DEPRECATED] bitsandbytes8 (LMI container versions >= 0.25.0 and < 0.27.0)
* gptq (LMI container versions >= 0.24.0)
* squeezellm (LMI container versions >= 0.27.0)

When using pre-quantized models make sure to use the correct model artifacts e.g. `TheBloke/Llama-2-13B-chat-GPTQ`, `TheBloke/Llama-2-13B-chat-AWQ`.

## Advanced LMI-Dist Configurations

Here are the advanced parameters that are available when using LMI-Dist.


| Item	                                    | LMI Version | Configuration Type	 | Description	                                                                                                                                                                                                                                                                                                                                                                                         | Example value	         |
|------------------------------------------|-------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| option.quantize	                         | \>= 0.23.0  | LMI	                | Quantize the model with the supported quantization methods(`gptq`, `awq`, `squeezellm`)                                                                                                                                                                                                                                                                              | `awq` Default: `None`	 |
| option.max_rolling_batch_prefill_tokens	 | \>= 0.24.0  | Pass Through	                | Limits the number of tokens for prefill(a.k.a prompt processing). This needs to be tuned based on GPU memory available and request lengths. Setting this value too high can limit the number of kv cache blocks or run into GPU OOM. If you don't set this, `lmi-dist` will default to max model length from Hugging Face config(also accounts for rope scaling if applicable).	                                                                                                                                                            | Default: `None`       |
| option.max_model_len                     | \>= 0.27.0  | Pass Through	       | The maximum length (input+output) of the request. The request will be stopped if more tokens are generated. `lmi-dist` will default to max model length from Hugging Face config(also accounts for rope scaling if applicable). For models with larger maximum length support(for e.g. 32k for Mistral 7B), it could lead to GPU OOM. In such cases, to deploy on a smaller instances, reduce this value.	 | Default: `None`	       |
| option.load_format	                      | \>= 0.27.0  | Pass Through	       | The checkpoint format of the model. Default is auto and means bin/safetensors will be used if found.	                                                                                                                                                                                                                                                                                                | Default: `auto`	       |
| option.enforce_eager                     | \>= 0.27.0  | Pass Through	       | `lmi-dist` by default will run with CUDA graph optimization to reach to the best performance. However, in the situation of very less GPU memory, having CUDA graph enabled will cause OOM. So if you set this option to true, we will use PyTorch Eager mode and disable CUDA graph to save some GBs of memory. `T5` model will not use cuda graphs. 	                                                                                           | Default: `False`	      |
| option.gpu_memory_utilization            | \>= 0.27.0  | Pass Through	       | This config controls the amount of GPU memory allocated to KV cache. Setting higher value will allocate more memory for KV cache. Default is 0.9. It recommended to reduce this value if GPU OOM's are encountered.                                                                                                                                                                                           | Default: `0.9`	        |
| option.speculative_draft_model                    | \>= 0.27.0  | Pass Through	       | Model id or path to speculative decoding draft model                                                                                                                                                                                                                                                                                                                                                       | Default: `None`	      |
| option.draft_model_tp_size                         | \>= 0.27.0  | Pass Through	       | Tensor parallel degree of speculative decoding draft model. Accepted values are `1` and target model's tensor parallel size(`option.tensor_parallel_degree`)                                                                                                                                                                                                                                                                     | Default: `1`	          |
| option.speculative_length                     | \>= 0.27.0  | Pass Through	       | Determines the number of tokens draft model generates before verifying against target model                                                                                                                                                                                                                                            | Default: `5`	         |
| option.record_acceptance_rate             | \>= 0.27.0  |  LMI	       | Enables logging speculative decoding acceptance rate                                                                                                                                                                                                                                                                                                   | Default: `False`	        |
| option.enable_lora                    | \>= 0.27.0  | Pass Through	       | This config enables support for LoRA adapters.                                                                                                                                                                                                                                                                                                                                                       | Default: `false`	      |
| option.max_loras                         | \>= 0.27.0  | Pass Through	       | This config determines the maximum number of LoRA adapters that can be run at once. Allocates GPU memory for those number adapters.                                                                                                                                                                                                                                                                    | Default: `4`	          |
| option.max_lora_rank                     | \>= 0.27.0  | Pass Through	       | This config determines the maximum rank allowed for a LoRA adapter. Set this value to maximum rank of your adapters. Setting a larger value will enable more adapters at a greater memory usage cost.                                                                                                                                                                                                                                                 | Default: `16`	         |
| option.lora_extra_vocab_size             | \>= 0.27.0  | Pass Through	       | This config determines the maximum additional vocabulary that can be added through a LoRA adapter.                                                                                                                                                                                                                                                                                                   | Default: `256`	        |
| option.max_cpu_loras                                | \>= 0.27.0  | Pass Through	       | This config determines the maximum number of LoRA adapters to cache in memory. All others will be evicted to disk.                                                                                                                                                                                                                                                                                   | Default: `None`	       |
