# Transformers-NeuronX Engine in LMI

## Model Artifacts Structure

LMI Transformers-NeuronX expects the model to be [standard HuggingFace format](../deployment_guide/model-artifacts.md) for runtime compilation. 

For loading of compiled models, both Optimum compiled models and split-models with a separate `neff` cache (compiled models must be compiled with the same Neuron compiler version and model settings).
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

## Quick Start Configurations

Most of the LMI Transformers-NeuronX models use the following template:

### serving.properties

```
engine=Python
option.model_id=<your model>
option.entryPoint=djl_python.transformers_neuronx
option.rolling_batch=auto
# Adjust the following based on model size and instance type
option.tensor_parallel_degree=4
option.max_rolling_batch_size=8
option.model_loading_timeout=1600
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---servingproperties) to deploy a model with serving.properties configuration on SageMaker.

### environment variables

```
HF_MODEL_ID=<your model>
OPTION_ENTRYPOINT=djl_python.transformers_neuronx
OPTION_ROLLING_BATCH=auto
# Adjust the following based on model size and instance type
TENSOR_PARALLEL_DEGREE=4
OPTION_MAX_ROLLING_BATCH_SIZE=8
OPTION_MODEL_LOADING_TIMEOUT=1600
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---environment-variables) to deploy a model with environment variable configuration on SageMaker.

## Quantization

Currently, we allow customer to use `option.quantize=static_int8` or `OPTION_QUANTIZE=static_int8` to load the model using `int8` weight quantization.

## Advanced Transformers NeuronX Configurations

Here are the advanced parameters that are available when using Transformers NeuronX.
Each advanced configuration is specified with a Configuration Type.
`LMI` means the configuration is processed by LMI and translated into the appropriate backend configurations.
`Pass Through` means the configuration is passed down directly to the library.
If you encounter an issue with a `Pass Through` configuration, it is likely an issue with the underlying library and not LMI.

| Item                                       | LMI Version | Configuration Type | Description                                                                                                                                                                                                                                                                                                                                           | Example value                                                                                  |
|--------------------------------------------|-------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| option.n_positions                         | >= 0.26.0   | Pass Through       | Total sequence length, input sequence length + output sequence length.                                                                                                                                                                                                                                                                                | Default: `128`                                                                                 |
| option.load_in_8bit                        | >= 0.26.0   | Pass Through       | Specify this option to quantize your model using the supported quantization methods in TransformerNeuronX                                                                                                                                                                                                                                             | `False`, `True` Default: `False`                                                               |
| option.unroll                              | >= 0.26.0   | Pass Through       | Unroll the model graph for compilation. With `unroll=None` compiler will have more opportunities to do optimizations across the layers                                                                                                                                                                                                                | Default: `None`                                                                                |
| option.neuron_optimize_level               | >= 0.26.0   | Pass Through       | Neuron runtime compiler optimization level, determines the type of optimizations applied during compilation. The higher optimize level we go, the longer time will spend on compilation. But in exchange, you will get better latency/throughput. Default value is not set (optimize level 2) that have a balance of compilation time and performance | `1`,`2`,`3` Default: `2`                                                                       |
| option.context_length_estimate             | >= 0.26.0   | Pass Through       | Estimated context input length for Llama models. Customer can specify different size bucket to increase the KV cache re-usability. This will help to improve latency                                                                                                                                                                                  | Example: `256,512,1024` (integers separated by comma if multiple values) <br/> Default: `None` |
| option.low_cpu_mem_usage                   | >= 0.26.0   | Pass Through       | Reduce CPU memory usage when loading models.                                                                                                                                                                                                                                                                                                          | Default: `False`                                                                               |
| option.load_split_model                    | >= 0.26.0   | Pass Through       | Toggle to True when using model artifacts that have already been split for neuron compilation/loading.                                                                                                                                                                                                                                                | Default: `False`                                                                               |
| option.compiled_graph_path                 | >= 0.26.0   | Pass Through       | Provide an s3 URI, or a local directory that stores the pre-compiled graph for your model (NEFF cache) to skip runtime compilation.                                                                                                                                                                                                                   | Default: `None`                                                                                |
| option.group_query_attention               | >= 0.26.0   | Pass Through       | Enable K/V cache sharding for llama and mistral models types  based on various [strategies](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html#grouped-query-attention-gqa-support-beta)                                                                                | `shard-over-heads`  Default: `None`                                                            |
| option.enable_mixed_precision_accumulation | >= 0.26.0   | Pass Through       | Turn this on for LLAMA 70B model to achieve better accuracy.                                                                                                                                                                                                                                                                                          | `true` Default: `None`                                                                         |

