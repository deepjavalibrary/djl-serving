# DeepSpeed Engine User Guide

## Model Artifacts Structure

DeepSpeed expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

The LMI 0.26.0 container currently ships DeepSpeed 0.12.6.
DeepSpeed supports two modes for launching models: Kernel Injection, and Auto Tensor-Parallelism.
Model architectures that support Kernel Injection are defined [here](https://github.com/microsoft/DeepSpeed/blob/v0.12.6/deepspeed/module_inject/replace_policy.py).
These model architectures have been optimized for inference through custom CUDA Kernel implementations.
Model architectures that support Auto Tensor-Parallelism are defined [here](https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/#supported-models).
Auto Tensor-Parallelism is used to host models across GPUs that do not have custom CUDA Kernel support.

When using LMI's integration with DeepSpeed, we will automatically apply kernel injection if the model architecture is supported.

The below model architectures have been carefully tested with LMI's DeepSpeed integrations:
* GPT2
* Bloom
* GPT-J
* GPT-Neo
* GPT-Neo-X
* OPT
* Stable Diffusion
* Llama

### Complete Model Set

<details>
  <summary>Expand this section to see the full set of model architectures supported by DeepSpeed.</summary>

Kernel Injection:
* GPT2
* Bert
* Bloom
* GPT-J
* GPT-Neo
* GPT-Neo-X
* OPT
* Megatron
* DistilBert
* Stable Diffusion
* Llama
* Llama2
* InternLM

Auto Tensor-Parallelism
* albert
* baichuan
* bert
* bigbird_pegasus
* bloom
* camembert
* codegen
* codellama
* deberta_v2
* electra
* ernie
* esm
* falcon
* glm
* gpt-j
* gpt-neo
* gpt-neox
* longt5
* luke
* llama
* llama2
* m2m_100
* marian
* mistral
* mpt
* mvp
* nezha
* openai
* opt
* pegasus
* perceiver
* plbart
* qwen
* reformer
* roberta
* roformer
* splinter
* starcode
* t5
* xglm
* xlm_roberta
* yoso
</details>

## Quick Start Configurations

You can leverage DeepSpeed with LMI using the following starter configurations:

### serving.properties

```
engine=MPI
option.entryPoint=djl_python.deepspeed
option.tensor_parallel_degree=max
option.rolling_batch=deepspeed
option.model_id=<your model id>
option.max_rolling_batch_size=64
```

### environment variables

```
SERVING_LOAD_MODELS=test::MPI=/opt/ml/model
OPTION_ENTRYPOINT=djl_python.deepspeed
OPTION_TENSOR_PARALLEL_DEGREE=max
OPTION_ROLLING_BATCH=deepspeed
OPTION_MODEL_ID=<your model id>
OPTION_MAX_ROLLING_BATCH_SIZE=64
```

You can use the [SageMaker deployment template](../README.md#using-the-sagemaker-python-sdk-to-deploy-your-first-model-with-lmi) to deploy the model with environment variables.

## Quantization Support

We support two methods of runtime quantization when using DeepSpeed with LMI: SmoothQuant, and Dynamic Int8.

If you are enabling quantization, we recommend using SmoothQuant for GPT2, GPT-J, GPT-Neo, GPT-Neo-X, Bloom, Llama model architectures.
You can enable SmoothQuant using `option.quantize=smoothquant` in serving.properties, or `OPTION_QUANTIZE=smoothquant` environment variable.
You can optionally specify `option.smoothquant_alpha` in serving.properties, or`OPTION_QUANTIZE=smoothquant` environment variable to control the quantization behavior.
We recommend that you do not provide this value and let LMI determine it.

For models not supported for SmoothQuant, you can enable Dynamic Int8 quantization using `option.quantize=dynamic_int8` in serving.properties, or `OPTION_QUANTIZE=dynamic_int8` environment variable.
The Dynamic Int8 quantization method uses DeepSpeed's [Mixture-of-Quantization](https://www.deepspeed.ai/tutorials/MoQ-tutorial/) algorithm for quantization.

## Advanced Configurations

Here are the advanced parameters that are available when using DeepSpeed


| Item	                      | Required	 | Description	                                                                                                                                                                                                                                                                                                                             | Example value	                   |
|----------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| option.task	               | No	       | The task used in Hugging Face for different pipelines. Default is text-generation	                                                                                                                                                                                                                                                       | `text-generation`	               |
| option.quantize	           | No	       | Specify this option to quantize your model using the supported quantization methods in DeepSpeed. SmoothQuant is our special offering to provide quantization with better quality	                                                                                                                                                       | `dynamic_int8`, `smoothquant`	   |
| option.max_tokens	         | No	       | Total number of tokens (input and output) with which DeepSpeed can work. The number of output tokens in the difference between the total number of tokens and the number of input tokens. By default we set the value to 1024. If you are looking for long sequence generation, you may want to set this to higher value (2048, 4096..)	 | 1024	                            |
| Advanced parameters	       |
| option.low_cpu_mem_usage	  | No	       | Reduce CPU memory usage when loading models. We recommend that you set this to True.	                                                                                                                                                                                                                                                    | Default:`true` 	                 |
| option.enable_cuda_graph	  | No	       | Activates capturing the CUDA graph of the forward pass to accelerate.	                                                                                                                                                                                                                                                                   | Default: `false`	                |
| option.triangular_masking	 | No	       | Whether to use triangular masking for the attention mask. This is application or model specific.	                                                                                                                                                                                                                                        | Default: `true`	                 |
| option.return_tuple	       | No	       | Whether transformer layers need to return a tuple or a tensor.	                                                                                                                                                                                                                                                                          | Default: `true`	                 |
| option.training_mp_size	   | No	       | If the model was trained with DeepSpeed, this indicates the tensor parallelism degree with which the model was trained. Can be different than the tensor parallel degree desired for inference.	                                                                                                                                         | Default: `1`	                    |
| option.checkpoint	         | No	       | Path to DeepSpeed compatible checkpoint file.	                                                                                                                                                                                                                                                                                           | `ds_inference_checkpoint.json`	  |
| option.smoothquant_alpha	  | No	       | If `smoothquant` is provided in option.quantize, you can provide this alpha value. If not provided, DeepSpeed will choose one for you.	                                                                                                                                                                                                  | Any float value between 0 and 1	 |
