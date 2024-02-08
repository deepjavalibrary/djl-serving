# vLLM Engine in User Guide

## Model Artifacts Structure

vLLM expect the model to be standard HuggingFace format.
The source of the model could be:

- model_id string from huggingface
- s3 url that store the artifact that follows the HuggingFace model repo structure
- A local path that contains everything in a folder follows HuggingFace model repo structure

### Standard HuggingFace model artifacts
Under the same folder level, we expect:
- config.json: Store the model architecture and structure information
- tokenizer_config.json: Store the tokenizer config information
- modelling files (*.py): If your model has custom modelling or tokenizer files.
  - Please remember to turn on `option.trust_remote_code=true` or `OPTION_TRUST_REMOTE_CODE=true`
- other files that needs to take care

A safe way is just to keep every files inside.

We don't support malformed model repo from HuggingFace or other source that does not contain config.json or typical model style
like GGUF format and more.

## Supported Model architecture

LMI is shipping vLLM 0.2.7 with 0.26.0 containers, 
so technically we support all LLM that [vLLM 0.2.7 support](https://github.com/vllm-project/vllm/tree/v0.2.7?tab=readme-ov-file#about).

The model architecture that we carefully tested for vLLM (in CI):
- LLAMA
- Falcon
- Mistral
- Mixtral
- GPT-NeoX
- LLAMA with AWQ

### Complete model set

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
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
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)

We will add more model support for the future versions to have them tested. Please feel free to [file us an issue](https://github.com/deepjavalibrary/djl-serving/issues/new/choose) for more model coverage in CI.

### Quantization

Currently, we allow customer to use `option.quantize=awq` or `OPTION_QUANTIZE=awq` to load an AWQ quantized model in VLLM.

We will have GPTQ supported for vLLM in the upcoming version.

## Deployment tutorial

Most of the vLLM model could fall under the following templates:

### serving.properties
You can deploy with a serving.properties:

```
engine=Python
option.tensor_parallel_degree=max
option.model_id=<your model>
option.max_rolling_batch_size=64
option.rolling_batch=vllm
```

This is the standard no-code experience DJL-Serving provided.

### All environment variables

You can also deploy without even providing any artifacts to run with LMI through specifying everything in ENV:

```
SERVING_LOAD_MODELS=test::Python=/opt/ml/model
OPTION_MODEL_ID=<your model>
OPTION_TENSOR_PARALLEL_DEGREE=max
OPTION_MAX_ROLLING_BATCH_SIZE=64
OPTION_ROLLING_BATCH=vllm
```

You can use [SageMaker deployment template](../README.md#using-the-sagemaker-python-sdk-to-deploy-your-first-model-with-lmi) to deploy the model with environment variables.

### Advanced parameters

Here are the advanced parameters in vLLM: 

| Item	                                    | LMI Version | Required	 | Description	                                                                                                                                                                                                                                                                                                                                                                                         | Example value	         |
|------------------------------------------|-------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| option.quantize	                         | \>= 0.26.0  | No	            | Quantize the model with the supported quantization methods	                                                                                                                                                                                                                                                                                                                                          | `awq` Default: `None`	 |
| option.max_rolling_batch_prefill_tokens	 | \>= 0.26.0  | No	               | Limits the number of tokens for caching. This needs to be tuned based on batch size and input sequence length to avoid GPU OOM. If you don't set, vLLM will try to find a good number to fit in	                                                                                                                                                                                                     | Default: `None`	       |
| option.max_model_len                     | \>= 0.26.0  | No	               | the maximum length (input+output) vLLM should preserve memory for. If not specified, will use the default length the model is capable in config.json. in verion like 0.27.0, sometimes model's maximum length could go to 32k (Mistral 7B) and way beyond the supported KV token size. In that case to deploy on a small instance, we need to adjust this value within the range of KV Cache limit.	 | Default: `None`	       |
| option.load_format	                      | \>= 0.26.0  | No	               | The checkpoint format of the model. Default is auto and means bin/safetensors will be used if found.	                                                                                                                                                                                                                                                                                                | Default: `auto`	       |
| option.enforce_eager                    | \>= 0.27.0  | No	       | vLLM by default will run with CUDA graph optimization to reach to the best performance. However, in the situation of very less GPU memory, having CUDA graph enabled will cause OOM. So if you set this option to true, we will use PyTorch Eager mode and disable CUDA graph to save some GBs of memory.	                                                                                           | Default: `False`	                 |
| option.gpu_memory_utilization            | \>= 0.27.0  | No	       | This config controls the percentage of memory to be allocated to PagedAttention. Default to 0.9 (90%). We don't recommend to change this value because this impact the overall GPU memory allocations.                                                                                                                                                                                               | Default: `0.9`	                   |

