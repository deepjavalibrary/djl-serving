# HuggingFace Accelerate User Guide

The HuggingFace Accelerate backend is only recommended when the model you are deploying is not supported by the other backends.
It is typically less performant than the other available options.
You should confirm that your model is not supported by other backends before using HuggingFace Accelerate.

## Model Artifact Structure

HuggingFace Accelerate expects the model to be in the [standard HuggingFace format](../deployment_guide/model-artifacts.md).

## Supported Model Architectures

LMI's HuggingFace Accelerate supports most models that are supported by HuggingFace Transformers.

For text-generation models (i.e. `*ForCausalLM`, `*LMHeadModel`, and `ForConditionalGeneration` architectures), LMI provides continuous batching support.
This significantly increases throughput compared to the base Transformers and Accelerate library implementations, and is the recommended operating mode for such models.

For non text-generation models, LMI will create the model via the Transformers [`pipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines) API.
These models are not compatible with continuous batching, and have not been extensively tested with LMI.

The set of supported tasks in LMI, and corresponding model architectures are:

* text-generation (`*ForCausalLM`, `*LMHeadModel`)
* text2text-generation (`*ForConditionalGeneration`)
* table-question-answering (`*TapasForQuestionAnswering`)
* question-answering (`*ForQuestionAnswering`)
* token-classification (`*ForTokenClassification`)
* text-classification (`*ForSequenceClassification`)
* multiple-choice (`*ForMultipleChoice`)
* fill-mask (`*ForMaskedLM`)

## Quick Start Configurations

You can leverage HuggingFace Accelerate with LMI using the following configurations:

### serving.properties

```
engine=Python
# use "scheduler" if deploying a text-generation model, and "disable" for other tasks (can also the config omit entirely)
option.rolling_batch=scheduler
option.model_id=<your model id>
# use max to partition the model across all gpus. This is naive sharding, where the model is sharded vertically (as opposed to horizontally with tensor parallelism)
option.tensor_parallel_degree=max
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---servingproperties) to deploy a model with serving.properties configuration on SageMaker.

### environment variables

```
HF_MODEL_ID=<your model id>
TENSOR_PARALLEL_DEGREE=max
OPTION_ROLLING_BATCH=scheduler
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#configuration---environment-variables) to deploy a model with environment variable configuration on SageMaker.

## Quantization Support

The HuggingFace Accelerate backend supports quantization via [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes).
Both 8-bit and 4-bit quantization are supported.

You can enable 8-bit quantization via `option.quantize=bitsandbytes8` in serving.properties, or `OPTION_QUANTIZE=bitsandbytes8` environment variable.
You can enable 4-bit quantization via `option.quantize=bitsandbytes4` in serving.properties, or `OPTION_QUANTIZE=bitsandbytes4` environment variable.

## Advanced HuggingFace Accelerate Configurations

The following table lists the advanced configurations that are available with the HuggingFace Accelerate backend.
There are two types of advanced configurations: `LMI`, and `Pass Through`.
`LMI` configurations are processed by LMI and translated into configurations that Accelerate uses.
`Pass Through` configurations are passed directly to the backend library. These are opaque configurations from the perspective of the model server and LMI.
We recommend that you file an [issue](https://github.com/deepjavalibrary/djl-serving/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=) for any issues you encounter with configurations.
For `LMI` configurations, if we determine an issue with the configuration, we will attempt to provide a workaround for the current released version, and attempt to fix the issue for the next release.
For `Pass Through` configurations it is possible that our investigation reveals an issue with the backend library.
In that situation, there is nothing LMI can do until the issue is fixed in the backend library.


| Item	                     | LMI Version | Configuration Type	 | Description	                                                                           | Example value	                    |
|---------------------------|-------------|---------------------|----------------------------------------------------------------------------------------|-----------------------------------|
| option.task	              | >= 0.25.0   | LMI                 | The task used in Hugging Face for different pipelines. Default is text-generation      | `text-generation`	                |
| option.quantize	          | >= 0.25.0	  | Pass Through        | Specify this option to quantize your model using the supported quantization methods. 	 | `bitsandbytes4`, `bitsandbytes8`	 |
| option.low_cpu_mem_usage	 | >= 0.25.0   | 	   Pass Through    | Reduce CPU memory usage when loading models. We recommend that you set this to True.	  | Default:`true` 	                  |
