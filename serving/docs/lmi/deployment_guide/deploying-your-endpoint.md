# Deploying your model on a SageMaker Endpoint

We recommend using the SageMaker Python SDK to deploy your model on SageMaker. 
Depending on which configuration format you are using (`serving.properties` file or environment variables), the steps are slightly different.
You will need the following to deploy your model with LMI on SageMaker:

* Model Artifacts (either HuggingFace Hub Model Id, or S3 URI pointing to model artifacts)
* Instance Type
* Container URI
* Configuration File or Environment Variables

## Option 1: Configuration - serving.properties

If your configuration is specified in a `serving.properties` file, we will need to upload this file to S3.
If your model artifacts are also stored in S3, then you can either upload the `serving.properties` file under the same object prefix, or in a separate location.
If you upload the `serving.properties` file under the same object prefix as your model artifacts, you should not specify `option.model_id` in the configuration.

For example, your directory containing model artifacts and serving.properties may look like this (flat structure):

```shell
my-lmi-model/
  - serving.properties
  - config.json
  - model-0001-of-0002.safetensors
  - model-0002-of-0002.safetensors
  - model.safetensors.index.json
  - tokenizer.json
  - tokenizer.model
```

If you upload the `serving.properties` to a different location than your model artifacts, make sure `option.model_id` points to the S3 URI of the object prefix (e.g. `option.model_id=s3://my-bucket/my-model-artifacts/`).

Here is the sample code you can use to upload your configuration to s3:

```shell
# Assumes that the serving.properties file is stored in a local folder called my-lmi-model/
# The my-lmi-model folder my also contain your model artifacts
aws s3 sync ./my-lmi-model s3://my-bucket/my-lmi-model 
```

Here is the code to deploy your model on SageMaker using LMI:

```python
import sagemaker

# Your IAM role that provides access to SageMaker and S3. 
# See https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html if running on a SageMaker notebook
iam_role = "my-sagemaker-role-arn"
# manages interactions with the sagemaker apis
sagemaker_session = sagemaker.session.Session()
# region is needed to retrieve the lmi container
region = sagemaker_session._region_name
# get the lmi image uri
# available frameworks: "djl-deepspeed" (for vllm, lmi-dist, deepspeed), "djl-tensorrtllm" (for tensorrt-llm), "djl-neuronx" (for transformers neuronx)
container_uri = sagemaker.image_uris.retrieve(framework="djl-deepspeed", version="0.27.0", region=region)
# create a unique endpoint name
endpoint_name = sagemaker.utils.name_from_base("my-lmi-endpoint")
# s3 uri object prefix under which the serving.properties and optional model artifacts are stored
# This is how we can specify uncompressed model artifacts
model_data = {
    "S3DataSource": {
        "S3Uri": "s3://my-bucket/my-lmi-model/",
        'S3DataType': 'S3Prefix',
        'CompressionType': 'None'
    }
} 
# instance type you will deploy your model to
instance_type = "ml.g5.12xlarge"

# create your SageMaker Model
model = sagemaker.Model(image_uri=container_uri, model_data=model_data, role=iam_role)
# deploy your model
model.deploy(
    instance_type=instance_type,
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)

# Get a predictor for your endpoint
predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)

# Make a prediction with your endpoint
outputs = predictor.predict({
    "inputs": "The meaning of life is", 
    "parameters": {"do_sample": True, "max_new_tokens": 256}
})
```

## Option 2: Configuration - environment variables

If you are using environment variables for configuration, you will need to pass those configurations to the SageMaker Model object.

```python
import sagemaker

# Your IAM role that provides access to SageMaker and S3. 
# See https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html if running on a SageMaker notebook
iam_role = "my-sagemaker-role"
# manages interactions with the sagemaker apis
sagemaker_session = sagemaker.session.Session()
# region is needed to retrieve the lmi container
region = sagemaker_session._region_name
# get the lmi image uri
# available frameworks: "djl-deepspeed" (for vllm, lmi-dist, deepspeed), "djl-tensorrtllm" (for tensorrt-llm), "djl-neuronx" (for transformers neuronx)
container_uri = sagemaker.image_uris.retrieve(framework="djl-deepspeed", version="0.27.0", region=region)
# create a unique endpoint name
endpoint_name = sagemaker.utils.name_from_base("my-lmi-endpoint")
# instance type you will deploy your model to
instance_type = "ml.g5.12xlarge"

# create your SageMaker Model
model = sagemaker.Model(
    image_uri=container_uri, 
    role=iam_role,
    # specify all environment variable configs in this map
    env={
        "HF_MODEL_ID": "<huggingface hub model id or s3 uri>",
        "OPTION_ROLLING_BATCH": "vllm",
        "TENSOR_PARALLEL_DEGREE": "max",
    }
)
# deploy your model
model.deploy(
    instance_type=instance_type,
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)

# Get a predictor for your endpoint
predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)

# Make a prediction with your endpoint
outputs = predictor.predict({
    "inputs": "The meaning of life is", 
    "parameters": {"do_sample": True, "max_new_tokens": 256}
})
```

Depending on which backend you are deploying with, you will have access to different generation parameters.
To learn more about the API schema (Request/Response structure), please see [this document](../user_guides/lmi_input_output_schema.md).

# Deploying your model on a SageMaker Endpoint with boto3

As an alternate to using the SageMaker python sdk, you could also use a sagemaker client from boto3. Currently, this provides with some additional options to configure your SageMaker endpoint.

## Using ProductionVariants to customize the endpoint
The following options may be added to the `ProductionVariants` field to support LLM deployment:

1. VolumeSizeInGB: option to attach a larger EBS volume if necessary.
2. ModelDataDownloadTimeoutInSeconds: some models, due to their large size, take longer to be downloaded from S3 and untarred. With this option, you can increase the timeout for such downloads.
3. ContainerStartupHealthCheckTimeoutInSeconds: Since LLMs may take longer to be loaded and ready to serve, the default /ping timeout of 5 minutes may not be sufficient. With this option, you can increase the timeout for the health check.

Follow this link for a step-by-step guide to create an endpoint with the above options: https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-hosting.html

## Using ModelDataSource to deploy uncompressed model artifacts
The following options may be added to the `ModelDataSource` field to support uncompressed artifacts from S3:
```
"ModelDataSource": {
            "S3DataSource": {
                "S3Uri": "s3://my-bucket/prefix/to/model/data/", 
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            },
```

Follow this link for a detailed overview of this option: https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-uncompressed.html

Next: [Benchmark your endpoint](benchmarking-your-endpoint.md)

Previous: [Container Configurations](configurations.md)