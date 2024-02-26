# Transformers-NeuronX Engine in LMI

## Model Artifacts Structure

LMI Transformers-NeuronX expect the model to be standard HuggingFace format for runtime compilation. For loading of compiled models, both Optimum compiled models and split-models with a separate `neff` cache (compiled models must be compiled with the same Neuron compiler version and model settings).
The source of the model could be:

- model_id string from huggingface
- s3 url that store the artifact that follows the HuggingFace model repo structure, the Optimum compiled model repo structure, or a split model with a second url for the `neff` cache.
- A local path that contains everything in a folder follows HuggingFace model repo structure, the Optimum compiled model repo structure, or a split model with a second directory for the `neff` cache.

More detail on the options for model artifacts available for the LMI Transformers-NeuronX container available [here](../deployment_guide/model-artifacts.md#neuron-pretrained-model-formats)

## Supported Model architecture

The model architectures that are tested daily for LMI Transformers-NeuronX (in CI):
- LLAMA
- Mistral
- GPT-NeoX
- GPT-J
- Bloom
- GPT2
- OPT

### Complete model set

- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)

We will add more model support for the future versions to have them tested. Please feel free to [file us an issue](https://github.com/deepjavalibrary/djl-serving/issues/new/choose) for more model coverage in CI.

### Quantization

Currently, we allow customer to use `option.quantize=static_int8` or `OPTION_QUANTIZE=static_int8` to load the model using `int8` weight quantization.

## Quick Start Configurations

Most of the LMI Transformers-NeuronX models use the following template (quantization, and other advanced features will require additional configuration which is described [here](../../lmi/configurations_large_model_inference_containers.md#transformers-neuronx-doc)):

### Use serving.properties
You can deploy with a serving.properties:

```
engine=Python
option.tensor_parallel_degree=4
option.model_id=<your model>
option.entryPoint=djl_python.transformers_neuronx
option.max_rolling_batch_size=8
option.rolling_batch=auto
option.model_loading_timeout=1600
```

This is the standard no-code experience DJL-Serving provided.

### Use environment variables

You can also deploy without even providing any artifacts to run with LMI through specifying everything in ENV:

```
SERVING_LOAD_MODELS=test::Python=/opt/ml/model
OPTION_MODEL_ID=<your model>
OPTION_ENTRYPOINT=djl_python.transformers_neuronx
OPTION_TENSOR_PARALLEL_DEGREE=4
OPTION_MAX_ROLLING_BATCH_SIZE=8
OPTION_ROLLING_BATCH=auto
OPTION_MODEL_LOADING_TIMEOUT=1600
```

### SageMaker notebook deployment samples

Here you can find deployment samples with SageMaker notebooks [tutorial](../README.md).