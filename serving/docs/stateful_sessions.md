# Stateful Sessions Support in LMI

Stateful sessions is a feature that allows all requests within the same session routed to the same instance, allowing your ML application to reuse previously processed information. This reduces latency and enhances the overall user experience.

## Stateful Sessions Configurations

* `OPTION_ENABLE_STATEFUL_SESSIONS`: Whether to enable stateful sessions support, defaults to true.
* `OPTION_SESSIONS_PATH`: Specifies the path where session data is saved, defaults to "/dev/shm/djl_sessions".
* `OPTION_SESSIONS_EXPIRATION`: Specifies time in seconds a session remains valid before it expires, defaults to 1200.

## Deploying with LMI

We configure the session expiration to 3600 seconds.

```
env = {
    "HF_MODEL_ID": "unsloth/llama-3-8b-Instruct",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_SESSIONS_EXPIRATION": "3600"
}

image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128"
model_name = name_from_base("llama-3-8b-instruct-stateful")

create_model_response = sm_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = {
        "Image": image_uri,
        "Environment": env,
    },
)

endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"

endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": "ml.g6.12xlarge",
            "InitialInstanceCount": 2,
            "ModelDataDownloadTimeoutInSeconds": 1800,
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
        },
    ],
)
create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

## Start Session

To start a session with a stateful model, send an `InvokeEndpoint` request. In the request payload, set "requestType" to "NEW_SESSION" to start a new session.

```
payload = {
    "requestType": "NEW_SESSION"
}
payload = json.dumps(payload)

create_session_response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=payload,
    ContentType="application/json",
    SessionId="NEW_SESSION")
```

The LMI container handles the request by starting a new session. The container provides the session ID and expiration timestamp (UTC timezone) by setting the following HTTP header in the response:

```
X-Amzn-SageMaker-Session-Id: session_id; Expires=yyyy-mm-ddThh:mm:ssZ
```

We can extract the session ID from the invoke_endpoint response.

```
session_id = create_session_response['ResponseMetadata']['HTTPHeaders']['x-amzn-sagemaker-new-session-id'].split(';')[0]
```

## Make Inference Requests

To use the same session for a subsequent inference request, the client sends another `InvokeEndpoint` request, specifying the session ID in the `SessionId` parameter. SageMaker platform then routes the request to the same ML instance where the session was started.

```
response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps({"inputs": "What is Amazon SageMaker?"}),
    ContentType="application/json",
    SessionId=session_id
)

response_model["Body"].read().decode("utf8")
```

## Close Session

To close a session, the client sends a final `InvokeEndpoint` request, providing the session ID in the `SessionId` parameter and setting "requestType" to "CLOSE" in the request payload.

```
payload = {
    "requestType": "CLOSE"
}
payload = json.dumps(payload)

close_session_response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=payload,
    ContentType="application/json",
    SessionId=session_id)
```

The container returns the session ID by setting the following HTTP header in the response:

```
X-Amzn-SageMaker-Closed-Session-Id: session_id
```

We can extract the closed session ID from the invoke_endpoint response.

```
closed_session_id = close_session_response['ResponseMetadata']['HTTPHeaders']['x-amzn-sagemaker-closed-session-id']
```

The full notebook is available [here](https://github.com/deepjavalibrary/djl-demo/blob/master/aws/sagemaker/large-model-inference/sample-llm/stateful_inference_llama3_8b.ipynb).
