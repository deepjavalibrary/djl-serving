# Container and Model Configurations

The configuration supplied to LMI provides required and optional information that LMI will use to load and serve your model.
LMI containers accept configurations provided in two formats. In order of priority, these are:

* `serving.properties` Configuration File (per model configurations)
* Environment Variables (global configurations)

We recommend using the `serving.properties` configuration file for the following reasons:
* Supports SageMaker Multi Model Endpoints with per model configurations 
  * Environment Variables are applied globally to all models hosted by the model server, so they can't be used for model specific configuration
* Separates model configuration from the SageMaker Model Object (deployment unit)
  * Configurations can be modified and updated independently of the deployment unit/code 

Environment Variables are a good option for the proof-of-concept and experimentation phase for a single model.
You can modify the environment variables as part of your deployment script without having to re-upload configurations to S3.
This typically leads to a faster iteration loop when modifying and experimenting with configuration values.

While you can mix configurations between `serving.properties` and environment variables, we recommend you choose one and specify all configuration in that format.
Configurations specified in the `serving.properties` files will override configurations specified in environment variables.

Both configuration mechanisms offer access to the same set of configurations.

If you know which backend you are going to use, you can find a set of starter configurations in the corresponding [user guide](../user_guides).
We recommend using the quick start configurations as a starting point if you have decided on a particular backend.
The only change required to the starter configurations is specifying `option.model_id` to point to your model artifacts.

We will now cover the components of a minimal starting configuration. This minimal configuration will look like:

```
# use standard python engine, or mpi aware python engine
engine=<Python|MPI>
# where the model artifacts are stored
option.model_id=<hf_hub_model_id|s3_uri>
# which inference library to use
option.rolling_batch=<auto|vllm|lmi-dist|deepspeed|tensorrtllm>
# how many gpus to shard the model across with tensor parallelism
option.tensor_parallel_degree=<max|number between 1 and number of gpus available>
```

There are additional configurations that can be specified.
We will cover the common configurations (across backends) in [LMI Common Configurations](#lmi-common-configurations)

## Model Artifact Configuration

If you are deploying a model hosted on the HuggingFace Hub, you must specify the `option.model_id=<hf_hub_model_id>` configuration.
When using a model directly from the hub, we recommend you also specify the model revision (commit hash) via `option.revision=<commit hash>`.
Since model artifacts are downloaded at runtime from the Hub, using a specific revision ensures you are using a model compatible with package versions in the runtime environment.
Open Source model artifacts on the hub are subject to change at any time, and these changes may cause issues when instantiating the model (the model may require a newer version of transformers than what is available in the container). 
If a model provides custom model (*modeling.py) and custom tokenizer (*tokenizer.py) files, you need to specify `option.trust_remote_code=true` to load and use the model.

If you are deploying a model hosted in S3, `option.model_id=<s3 uri>` should be the s3 object prefix of the model artifacts.
Alternatively, you can deploy the model artifacts alongside your `serving.properties` configuration file via the `model.tar.gz` artifact and not specify `option.model_id`.
SageMaker will extract the contents of the `model.tar.gz` archive to the default directory LMI uses to load and serve models.

## Inference Library Configuration

LMI expects the following two configurations to determine which backend to use:

* `engine`. The options are `Python` and `MPI`, which dictates how we launch the Python processes
* `option.rolling_batch`. This represents the inference library to use. The available options depend on the container.
* `option.entryPoint`. This represents the default inference handler to use. In most cases, this can be auto-detected and does not need to be specified

In the DeepSpeed Container:
* to use vLLM, use `engine=Python` and `option.rolling_batch=vllm`
* to use lmi-dist, use `engine=MPI` and `option.rolling_batch=lmi-dist`
* to use DeepSpeed, use `engine=MPI`, `option.rolling_batch=deepspeed`, and `option.entryPoint=djl_python.deepspeed`

In the TensorRT-LLM Container:
* use `engine=MPI` and `option.rolling_batch=trtllm` to use TensorRT-LLM

In the Transformers NeuronX Container:
* use `engine=Python` and `option.rolling_batch=auto` to use Transformers NeuronX

## Tensor Parallelism Configuration

The `option.tensor_parallel_degree` configuration is used to specify how many GPUs to shard the model across using tensor parallelism.
This value should be between 1, and the maximum number of GPUs available on an instance.

We recommend setting this value to `max`, which will shard your model across all available GPUs.

Alternatively, if this value is specified as a number, LMI will attempt to maximize the number of model workers based on available GPUs.

For example, using an instance with 4 gpus and a tensor parallel degree of 2 will result in 2 model copies, each using 2 gpus.

## LMI Common Configurations

There are two classes of configurations provided by LMI:

* Model Server level configurations. These configurations do not have a prefix (e.g. `job_queue_size`)
* Engine/Backend level configurations. These configurations have a `option.` prefix (e.g. `option.model_id`)

Since LMI is built using the DJLServing model server, all DJLServing configurations are available in LMI.
You can find a list of these configurations [here](https://docs.djl.ai/docs/serving/serving/docs/configurations_model.html#python-model-configuration).

The following list of configurations is intended to highlight the relevant configurations for LMI:

* `engine`: The runtime engine of the code. For LMI, the options are `MPI` or `Python`.
  * `MPI` should be used for mpi aware backends. Refer to the [user guides](../user_guides) to see whether a backend supports mpi or not
* `job_queue_size`: The job/request queue size for the model. After this limit is reached, requests will be rejected until capacity becomes available
  * Default: `1000`
* `option.model_id`: The HuggingFace Hub Model Id (e.g. meta-llama/Llama-2-13b-hf), or the S3 URI object prefix pointing to self-host model artifacts (e.g. s3://my-model-bucket/llama2-13b/)
* `option.trust_remote_code`: Set to `true` to use a HuggingFace Hub Model that contains custom code (*modeling.py or *tokenizer.py files)
  * Default: `false`
* `option.revision`: The commit hash of a HuggingFace Hub Model Id. We recommend setting this value to ensure you use a specific version of the model artifacts
* `option.rolling_batch`: Enables continuous batching (iteration level batching) with one of the supported backends. This is our recommended way of enabling a specific stack due to the significant performance improvements continuous batching brings at high concurrency levels
  * Possible Values:
    * All Containers: `auto`, which defers the backend decision to LMI. This decision is based on the model architecture
    * DeepSpeed Container: `auto`, `vllm`, `lmi-dist`, `deepspeed`
    * TensorRT-LLM Container: `auto`, `trtllm`
    * Transformers NeuronX Container: `auto`
* `option.max_rolling_batch_size`: The maximum number of requests/sequences the model can process at a time. This parameter should be tuned to maximize throughput while staying within the available memory limits. `job_queue_size` should be set to a value equal or higher to this value. If the current batch is full, new requests will be queued until they can be processed.
  * Default: `32` for all backends except DeepSpeed. For DeepSpeed, the default is `4` 
* `option.dtype`: The data type you plan to cast the model weights to
  * Default: `fp16`
  * Possible Values: `fp32`, `fp16`, `bf16` (only available on G5/P4/P5 and newer instance types), `int8` (only for `lmi-dist` backend)
* `option.tensor_parallel_degree`: The number of GPUs to shard the model across. 
  * Recommended Value: `max` (use all GPUS)
  * Default: 
    * DeepSpeed and Transformers NeuronX Containers: `1`
    * TensorRT-LLM Container: `max`
* `option.entryPoint`: The python script to use for loading the model and executing inference. We recommend using one of our built-in default handlers. You can also specify a custom script by providing the script name (e.g. `model.py`)
  * Default: This is auto-detected by LMI based on which backend you are using
  * Possible Values:
    * `djl_python.huggingface`: used for vLLM, lmi-dist, and huggingface accelerate (fallback)
    * `djl_python.deepspeed`: used for deepspeed
    * `djl_python.transformers_neuronx`: used for transformers neuronx and optimum neuron
    * `djl_python.tensorrt_llm`: used for tensorrt-llm
    * your own custom script 
* `option.parallel_loading`: If using multiple workers (multiple model copies), setting to `true` will load the model workers in parallel. This should only be set to `true` if using multiple model copies, and there is sufficient CPU memory to load N copies of the model (memory at least N * model size in GB)
  * Default: `false`
* `option.model_loading_timeout`: The maximum time for model loading before the model server times out.
  * Default: `1800` seconds (30 minutes)
  * Note: On SageMaker, if you set this value, you should also set `container_startup_health_check_timeout=<at least model loading timeout>`
* `option.output_formatter`: Defines the output format (response format) that the model server returns to the client
  * Default: `json`
  * Possible Values: `json`, `jsonlines`

## Backend Specific Configurations

Each backend provides access to additional configurations.
You can find these configurations in the respective [user guides](../user_guides).

## Environment Variable Configurations

All LMI Configuration keys available in the `serving.properties` format can be specified as environment variables.

The translation for `engine` is unique. The configuration `engine=<engine>` is translated to `SERVING_LOAD_MODELS=test::<engine>=/opt/ml/model`.
For example:

* `engine=Python` is translated to environment variable `SERVING_LOAD_MODELS=test::Python=/opt/ml/model`
* `engine=MPI` is translated to environment variable `SERVING_LOAD_MODELS=test::MPI=/opt/ml/model`

Configuration keys that start with `option.` can be specified as environment variables using the `OPTION_` prefix.
The configuration `option.<property>` is translated to environment variable `OPTION_<PROPERTY>`. For example:
 
* `option.model_id` is translated to environment variable `OPTION_MODEL_ID`
* `option.tensor_parallel_degree` is translated to environment variable `OPTION_TENSOR_PARALLEL_DEGREE`

Configuration keys that do not start with option can be specified as environment variables using the `SERVING_` prefix.
The configuration `<property>` is translated to environment variable `SERVING_<PROPERTY>`. For example:

* `job_queue_size` is translated to environment variable `SERVING_JOB_QUEUE_SIZE`

For a full example, given the following `serving.properties` file:

```
engine=MPI
option.model_id=tiiuae/falcon-40b
option.task=text-generation
option.entryPoint=djl_python.transformersneuronx
option.trust_remote_code=true
option.tensor_parallel_degree=4
option.max_rolling_batch_size=32
option.rolling_batch=lmi-dist
option.dtype=fp16
```

We can translate the configuration to environment variables like this:

```
SERVING_LOAD_MODELS=test::MPI=/opt/ml/model
OPTION_MODEL_ID=tiiuae/falcon-40b
OPTION_TASK=text-generation
OPTION_ENTRYPOINT=djl_python.transformersneuronx
OPTION_TRUST_REMOTE_CODE=true
OPTION_TENSOR_PARALLEL_DEGREE=4
OPTION_MAX_ROLLING_BATCH_SIZE=32
OPTION_ROLLING_BATCH=lmi-dist
OPTION_DTYPE=FP16
```

Next: [Deploying your endpoint](deploying-your-endpoint.md)
Previous: [Backend Selection](backend-selection.md)