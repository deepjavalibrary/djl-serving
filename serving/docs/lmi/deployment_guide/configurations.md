# Container and Model Configurations

The configuration supplied to LMI provides information that LMI will use to load and serve your model.
LMI containers accept configurations provided in two formats.

* `serving.properties` Configuration File (per model configurations)
* Environment Variables (global configurations)

For most use-cases, using environment variables is sufficient.
If you are deploying LMI to serve multiple models within the same container (SageMaker Multi-Model Endpoint use-case), you should use per model `serving.properties` configuration files.
Environment Variables are global settings and will apply to all models being served within a single instance of LMI.

While you can mix configurations between `serving.properties` and environment variables, we recommend you choose one and specify all configuration in that format.
Configurations specified in the `serving.properties` files will override configurations specified in environment variables.

Both configuration mechanisms offer access to the same set of configurations.

If you know which backend you are going to use, you can find a set of starter configurations in the corresponding [user guide](../user_guides/README.md).
We recommend using the quick start configurations as a starting point if you have decided on a particular backend.

We will now cover the two types of configuration formats

## serving.properties 

### Model Artifact Configuration (required)

If you are deploying model artifacts directly with the container, LMI will detect the artifacts in the default model store `/opt/ml/model`.
This is the default location when using SageMaker, and where SageMaker will mount the artifacts when specified via [`ModelDataSource`](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-uncompressed.html).
You do not need to set any model artifact configurations when using this mechanism.

If you are deploying a model hosted on the HuggingFace Hub, you must specify the `option.model_id=<hf_hub_model_id>` configuration.
When using a model directly from the hub, we recommend you also specify the model revision (commit hash or branch) via `option.revision=<commit hash/branch>`.
Since model artifacts are downloaded at runtime from the Hub, using a specific revision ensures you are using a model compatible with package versions in the runtime environment.
Open Source model artifacts on the hub are subject to change at any time. 
These changes may cause issues when instantiating the model (updated model artifacts may require a newer version of a dependency than what is bundled in the container). 
If a model provides custom model (*modeling.py) and/or custom tokenizer (*tokenizer.py) files, you need to specify `option.trust_remote_code=true` to load and use the model.

If you are deploying a model hosted in S3, `option.model_id=<s3 uri>` should be the s3 object prefix of the model artifacts.
Alternatively, you can upload the `serving.properties` file to S3 alongside your model artifacts (under the same prefix) and omit the `option.model_id` config from your `serving.properties` file.
Example code for leveraging uncompressed artifacts in S3 are provided in the [deploying your endpoint](deploying-your-endpoint.md#option-1-configuration---servingproperties) section.

### Inference Library Configuration (optional)

Inference library configurations are optional, but allow you to override the default backend for your model.
To override, or explicitly set the inference backend, you should set `option.rolling_batch`. 
This represents the inference library to use. 
The available options depend on the container.

In the LMI Container:

* to use vLLM, use `option.rolling_batch=vllm`
* to use lmi-dist, use `option.rolling_batch=lmi-dist`
* to use huggingface accelerate, use `option.rolling_batch=auto` for text generation models, or `option.rolling_batch=disable` for non-text generation models.

In the TensorRT-LLM Container:

* use `option.rolling_batch=trtllm` to use TensorRT-LLM (this is the default)

In the Transformers NeuronX Container:

* use `option.rolling_batch=auto` to use Transformers NeuronX (this is the default)

### Tensor Parallelism Configuration

The `option.tensor_parallel_degree` configuration is used to specify how many GPUs to shard the model across using tensor parallelism.
This value should be between 1, and the maximum number of GPUs available on an instance.

We recommend setting this value to `max`, which will shard your model across all available GPUs.

Alternatively, if this value is specified as a number, LMI will attempt to maximize the number of model workers based on available GPUs.

For example, using an instance with 4 gpus and a tensor parallel degree of 2 will result in 2 model copies, each using 2 gpus.


### LMI Common Configurations

There are two classes of configurations provided by LMI:

* Model Server level configurations. These configurations do not have a prefix (e.g. `job_queue_size`)
* Engine/Backend level configurations. These configurations have a `option.` prefix (e.g. `option.dtype`)

Since LMI is built using the DJLServing model server, all DJLServing configurations are available in LMI.
You can find a list of these configurations [here](../../configurations_model.md#python-model-configuration).

The following list of configurations is intended to highlight the relevant configurations for LMI: 

| Configuration Key             | Description                                                                                                                                                                                                                                                                                                                                       | Default Value                                                                                                                                                          | Possible Values / Examples                                                                                                                                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| engine                        | The runtime engine of the inference code. For LMI this should either be `MPI` or `Python` depending on which inference library you use                                                                                                                                                                                                            | `Python`                                                                                                                                                               | `Python`, `MPI`                                                                                                                                                                                                      |
| job_queue_size                | The job/request queue size for the model. After this limit is reached, requests will be rejected until capacity becomes available                                                                                                                                                                                                                 | 1000                                                                                                                                                                   | Integer                                                                                                                                                                                                              |
| option.model_id               | The HuggingFace Hub Model Id, or the S3 URI object prefix pointing to self-host model artifacts                                                                                                                                                                                                                                                   | None                                                                                                                                                                   | `meta-llama/Llama-2-7b-hf`, `s3://model-bucket/models/llama2-7b`                                                                                                                                                     |
| option.trust_remote_code      | Set to `true` to use a HuggingFace Hub Model that contains custom code (*modeling.py or *tokenizer.py files)                                                                                                                                                                                                                                      | `false`                                                                                                                                                                | `true`, `false`                                                                                                                                                                                                      |
| option.revision               | The commit hash of a HuggingFace Hub Model Id. We recommend setting this value to ensure you use a specific version of the model artifacts                                                                                                                                                                                                        | None                                                                                                                                                                   | `dc1d3b3bfdb69df26f8fc966c16353274b138c56`                                                                                                                                                                           |
| option.rolling_batch          | Enables continuous batching (iteration level batching) with one of the supported backends. Available backends differ by container, see [Inference Library Configurations](#inference-library-configuration-optional) for mappings                                                                                                                 | None                                                                                                                                                                   | `auto`, `vllm`, `lmi-dist`, `trtllm`                                                                                                                                                                                 |
| option.max_rolling_batch_size | The maximum number of requests/sequences the model can process at a time. This parameter should be tuned to maximize throughput while staying within the available memory limits. `job_queue_size` should be set to a value equal or higher to this value. If the current batch is full, new requests will be queued until they can be processed. | `256`                                                                                                                                                                  | Integer                                                                                                                                                                                                              |
| option.dtype                  | The data type you plan to cast the model weights to. If not provided, LMI will use fp16.                                                                                                                                                                                                                                                          | `fp16`                                                                                                                                                                 | `fp32`, `fp16`, `bf16` (only on G5/P4/P5 or newer instance types), `int8` (only in lmi-dist)                                                                                                                         | 
| option.tensor_parallel_degree | The number of GPUs to shard the model across. Recommended value is `max`, which partitions the model across all available GPUS                                                                                                                                                                                                                    | `1` for LMI and Transformers NeuronX containers, `max` for TensorRT-LLM container                                                                                      | Value between 1 and number of available GPUs. For Inferentia, this represents the number of neuron cores                                                                                                             | 
| option.entryPoint             | The inference handler to use. This is either one of the built-in handlers provided by lmi, or the name of a custom script provided to LMI                                                                                                                                                                                                         | `djl_python.huggingface` for LMI Container, `djl_python.tensorrt_llm` for TensorRT-LLM container, `djl_python.transformers_neuronx` for Transformers NeuronX container | `djl_python.huggingface` (vllm, lmi-dist, hf-accelerate), `djl_python.tensorrt_llm` (tensorrt-llm), `djl_python.transformers_neuronx` (transformers neuronx / optimum neuron), `<custom_script>.py` (custom handler) |
| option.parallel_loading       | If using multiple workers (multiple model copies), setting to `true` will load the model workers in parallel. This should only be set to `true` if using multiple model copies, and there is sufficient CPU memory to load N copies of the model (memory at least N * model size in GB)                                                           | `false`                                                                                                                                                                | `true`, `false`                                                                                                                                                                                                      |
| option.model_loading_timeout  | The maximum time for model loading before the model server times out. On SageMaker, if you set this value, you should also set `container_startup_health_check_timeout=<at least model loading timeout>`                                                                                                                                          | `1800` (seconds, 30 minutes)                                                                                                                                           | Integer representing time in seconds                                                                                                                                                                                 |
| option.output_formatter       | Defines the output format (response format) that the model server returns to the client                                                                                                                                                                                                                                                           | `json`                                                                                                                                                                 | `json`, `jsonlines`                                                                                                                                                                                                  | 


## Backend Specific Configurations

Each backend provides access to additional configurations.
You can find these configurations in the respective [user guides](../user_guides/README.md).

## Environment Variable Configurations

The core configurations available via environment variables are documented in our [starting guide](../user_guides/starting-guide.md#available-environment-variable-configurations).

For other configurations, the `serving.property` configuration can be translated into an equivalent environment variable configuration.

Keys that start with `option.` can be specified as environment variables using the `OPTION_` prefix.
The configuration `option.<property>` is translated to environment variable `OPTION_<PROPERTY>`. For example:
 
* `option.rolling_batch` is translated to environment variable `OPTION_ROLLING_BATCH`

Configuration keys that do not start with `option` can be specified as environment variables using the `SERVING_` prefix.
The configuration `<property>` is translated to environment variable `SERVING_<PROPERTY>`. For example:

* `job_queue_size` is translated to environment variable `SERVING_JOB_QUEUE_SIZE`

Next: [Deploying your endpoint](deploying-your-endpoint.md)

Previous: [Backend Selection](backend-selection.md)
