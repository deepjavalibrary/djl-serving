# LMI-Dist Engine User Guide

## Model Artifact Structure

LMI-Dist expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

The model architecture that we test for lmi-dist (in CI):

- Llama (un-quantized and with GPTQ)
- Falcon
- GPT-NeoX (un-quantized and with BitsAndBytes)
- MPT
- Mistral (un-quantized and with AWQ)
- Mixtral
- T5

### Complete Model Set

Optimized models:

* falcon
* gpt-neox
* llama
* llama-2
* llava
* mistral
* mixtral
* mpt
* santacoder
* t5

*Note: PEFT is also supported for optimized models*

`lmi-dist` can also run other models which are supported by huggingface transformers but not listed above, although **without** the optimizations and `gptq`, `awq` quantization support. 
`transformers` library support can be found for [CausalLM](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L381) and [Seq2Seq](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L652) models.

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

When using pre-quantized models make sure to use the correct model artifacts e.g. `TheBloke/Llama-2-13B-chat-GPTQ`, `TheBloke/Llama-2-13B-chat-AWQ`.

## Advanced LMI-Dist Configurations

Here are the advanced parameters that are available when using LMI-Dist.


| Item                                                              | LMI Version | Description                                                                                                                                                                                                                 | Example value   |
|-------------------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| option.quantize                                                   | \>= 0.23.0  | Quantize the model with the supported quantization methods                                                                                                                                                                  | Default: `None` |
| option.max_rolling_batch_prefill_tokens [Deprecated since 0.25.0] | \>= 0.24.0  | Limits the number of tokens for caching. This needs to be tuned based on batch size and input sequence length to avoid GPU OOM. Currently we are calculating the best value for you from 0.25.0, this is no longer required | Default: 4096   |

