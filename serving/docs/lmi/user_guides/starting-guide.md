# LMI Starting Guide

Most models can be served using the single `HF_MODEL_ID=<model_id>` environment variable.
However, some models require additional configuration.
You can refer to our example notebooks [here](https://github.com/deepjavalibrary/djl-demo/tree/master/aws/sagemaker/large-model-inference/sample-llm) for model-specific examples.

If you are unable to deploy a model using just `HF_MODEL_ID`, and there is no example in the notebook repository, please cut us a Github issue so we can investigate and help.

The following code example demonstrates this configuration UX using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

This example will use the [Llama 3.1 8b Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model. 

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker.djl_inference import DJLModel 

# Setup role and sagemaker session
iam_role = sagemaker.get_execution_role() 
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

# Create the SageMaker Model object. In this example we let LMI configure the deployment settings based on the model architecture  
model = DJLModel(
  model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
  role=iam_role,
  env={
    "HF_TOKEN": "<hf token value for gated models>",
  }
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
endpoint_name = sagemaker.utils.name_from_base("llama-8b-endpoint")
predictor = model.deploy(instance_type="ml.g5.12xlarge", initial_instance_count=1, endpoint_name=endpoint_name)

# Make an inference request against the llama2-7b endpoint
outputs = predictor.predict({
  "inputs": "The diamondback terrapin was the first reptile to be",
  "parameters": {
    "do_sample": True,
    "max_new_tokens": 256,
  }
})
```

## Supported Model Architectures

If you are deploying with the LMI container (e.g. `763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124`), you can find the list of supported models [here](lmi-dist_user_guide.md#supported-model-architectures).

If you are deploying with the LMI-TRT container (e.g. `763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.30.0-tensorrtllm0.12.0-cu125`), you can find the list of supported models [here](trt_llm_user_guide.md#supported-model-architectures).

If you are deploying with the LMI-Neuron container (e.g. `763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1`), you can find the list of supported models [here](tnx_user_guide.md#supported-model-architecture).

## Available Environment Variable Configurations

The following environment variables are exposed as part of this simplified UX:

**HF_MODEL_ID**

This configuration is used to specify the location of your model artifacts.
It can either be a HuggingFace Hub model-id (e.g.meta-llama/Meta-Llama-3.1-8B-Instruct), a S3 uri (e.g. s3://my-bucket/my-model/), or a local path.
If you are using [SageMaker's capability to specify uncompressed model artifacts](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-uncompressed.html), you should set this value to `/opt/ml/model`.
`/opt/ml/model` is the path in the container where model artifacts are mounted if using this mechanism.

**HF_REVISION**

If you are using a model from the HuggingFace Hub, this specifies the commit or branch to use when downloading the model.

This is an optional config, and does not have a default value. 

**HF_TOKEN**

Some models on the HuggingFace Hub are gated and require permission from the owner to access.
To deploy a gated model from the HuggingFace Hub using LMI, you must provide an Access Token via this environment variable.

**HF_MODEL_TRUST_REMOTE_CODE**

If the model artifacts contain custom modeling code, you should set this to true after validating the custom code is not malicious.
If you are using a HuggingFace Hub model id, you should also specify `HF_REVISION` to ensure you are using artifacts and code that you have validated.

This is an optional config, and defaults to `False`.

**TENSOR_PARALLEL_DEGREE**

This value is used to specify the number of GPUs to partition the model across using tensor parallelism.
The value should be less than or equal to the number of available GPUs on the instance type you are deploying with.
We recommend setting this to `max`, which will shard the model across all available GPUs.

This is an optional config, and defaults to `max`.

**MAX_BATCH_SIZE**

This value represents the maximum number of requests the model will handle in a batch.

This is an optional config, and defaults to `256`.

**MAX_CONCURRENT_REQUESTS**

This value represents the number of active requests/client connections the model server can handle.
This value should be greater than or equal to `MAX_BATCH_SIZE`. 
For requests received when a full batch is being executed, they will be queued until a free slot in the batch becomes available.

This is an optional config, and defaults to `1000`.

### Additional Configurations

Additional configurations are available to further tune and customize your deployment.
These configurations are covered as part of the advanced deployment guide [here](../deployment_guide/configurations.md).
In most cases, the simplified configuration set described here is sufficient to achieve high performance.
We recommend you first deploy using the configurations described here, and venture into the advanced configurations if you need additional customizations and performance tuning.

The configurations described in that doc follow a different naming convention when using the `serving.properties` configuration format.
The above environment variables translate to the following `serving.properties` configurations:

* `HF_MODEL_ID`: `option.model_id`
* `HF_REVISION`: `option.revision`
* `HF_MODEL_TRUST_REMOTE_CODE`: `option.trust_remote_code`
* `TENSOR_PARALLEL_DEGREE`: `option.tensor_parallel_degree`
* `MAX_BATCH_SIZE`: `option.max_rolling_batch_size`
* `MAX_CONCURRENT_REQUESTS`: `job_queue_size`

## API Schema

The request and response schema for interacting with the model endpoint is available [here](lmi_input_output_schema.md).
