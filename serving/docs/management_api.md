# DJL Serving Management API

DJL Serving provides a set of API allow user to manage models at runtime:

1. [Register a model](#register-a-model)
2. [Increase/decrease number of workers for specific model](#scale-workers)
3. [Describe a model's status](#describe-model-or-workflow)
4. [Unregister a model](#unregister-a-model-or-workflow)
5. [List registered models](#list-workflows)

In addition, there is also the [adapter management API](adapters_api.md) for managing adapters.

Management API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [DJL Serving Configuration](configuration.md).

Similar as [Inference API](inference_api.md).

## Management APIs

### Register a model

Registers a new model as a single model workflow. The workflow name and version matches the model name and version.

`POST /models`

* url - Model url.
* model_name - the name of the model and workflow; this name will be used as {workflow_name} in other API as path.
  If this parameter is not present, modelName will be inferred by url.
* model_version - the version of the mode
* engine - the name of engine to load the model. DJL will try to infer engine if not specified.
* device - the device to load the model. DJL will pick optimal device if not specified, the value device can be:
    * CPU device: cpu or simply -1
    * GPU device: gpu0, gpu1, ... or simply 0, 1, 2, 3, ...
    * Neuron core: nc1, nc2, ...
* job_queue_size: the request job queue size, default is `1000`.
* batch_size - the inference batch size, default is `1`.
* max_batch_delay - the maximum delay for batch aggregation in millis, default value is `100` milliseconds.
* max_idle_time - the maximum idle time in seconds before the worker thread is scaled down, default is `60` seconds.
* min_worker - the minimum number of worker processes, DJL will auto detect minimum workers if not specified.
* max_worker - the maximum number of worker processes, DJL will auto detect maximum workers if not specified.
* synchronous - if the creation of worker is synchronous, the default is `true`.

```bash
curl -X POST "http://localhost:8080/models?url=https%3A%2F%2Fresources.djl.ai%2Ftest-models%2Fmlp.zip"

{
  "status": "Model \"mlp\" registered."
}
```

Download and load model may take some time, user can choose asynchronous call and check the status later.

The asynchronous call will return before trying to create workers with HTTP code 202:

```bash
curl -v -X POST "http://localhost:8080/models?url=https%3A%2F%2Fresources.djl.ai%2Ftest-models%2Fmlp.zip&synchronous=false"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: bf998daa-892f-482b-a660-6d0447aa5a7a
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 56
< connection: keep-alive
< 
{
  "status": "Model \"mlp\" registration scheduled."
}
```

### Register a workflow

`POST /workflows`

* url - Workflow url.
* engine - the name of engine to load the model. DJL will try to infer engine if not specified.
* device - the device to load the model. DJL will pick optimal device if not specified, the value device can be:
    * CPU device: cpu or simply -1
    * GPU device: gpu0, gpu1, ... or simply 0, 1, 2, 3, ...
    * Neuron core: nc1, nc2, ...
* min_worker - the minimum number of worker processes. The default value is `1`.
* max_worker - the maximum number of worker processes. The default is the same as the setting for `min_worker`.
* synchronous - if the creation of worker is synchronous. The default value is true.

```bash
curl -X POST "http://localhost:8080/workflows?url=https%3A%2F%2Fresources.djl.ai%2Ftest-workflows%2Fmlp.zip"

{
  "status": "Workflow \"mlp\" registered."
}
```

Download and load model may take some time, user can choose asynchronous call and check the status later.

The asynchronous call will return before trying to create workers with HTTP code 202:

```bash
curl -v -X POST "http://localhost:8080/workflows?url=https%3A%2F%2Fresources.djl.ai%2Ftest-workflows%2Fmlp.zip&synchronous=false"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: bf998daa-892f-482b-a660-6d0447aa5a7a
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 56
< connection: keep-alive
< 
{
  "status": "Workflow \"mlp\" registration scheduled."
}
```

### Scale workers

`PUT /models/{model_name}`

`PUT /models/{model_name}/{version}`

`PUT /workflows/{workflow_name}`

`PUT /workflows/{workflow_name}/{version}`

* min_worker - the minimum number of worker processes. The default value is `1`.
* max_worker - the maximum number of worker processes. The default is the same as the setting for `min_worker`.
* synchronous - if the creation of worker is synchronous. The default value is true.

Use the Scale Worker API to dynamically adjust the number of workers to better serve different inference request loads.

There are two different flavour of this API, synchronous vs asynchronous.

The asynchronous call will return immediately with HTTP code 202:

```bash
curl -v -X PUT "http://localhost:8080/workflows/mlp?min_worker=3&synchronous=false"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 74b65aab-dea8-470c-bb7a-5a186c7ddee6
< content-length: 33
< connection: keep-alive
< 
{
  "status": "Worker updated"
}
```

### Describe model or workflow

`GET /models/{model_name}`

`GET /workflows/{workflow_name}`

Use the Describe Model API to get detail runtime status of a model or workflow:

```bash
curl http://localhost:8080/models/mlp

[
  {
    "workflowName": "mlp",
    "models": [
      {
        "modelName": "mlp",
        "modelUrl": "https://resources.djl.ai/test-models/mlp.zip",
        "batchSize": 1,
        "maxBatchDelayMillis": 100,
        "maxIdleSeconds": 60,
        "queueSize": 1000,
        "requestInQueue": 0,
        "status": "Healthy",
        "loadedAtStartup": true,
        "workerGroups": [
          {
            "device": {
              "deviceType": "cpu",
              "deviceId": -1
            },
            "minWorkers": 1,
            "maxWorkers": 12,
            "workers": [
              {
                "id": 1,
                "startTime": "2023-06-08T08:14:16.999Z",
                "status": "READY"
              }
            ]
          }
        ]
      }
    ]
  }
]
```

### Unregister a model or workflow

`DELETE /models/{model_name}`

`DELETE /workflows/{workflow_name}`

Use the Unregister Model or workflow API to free up system resources:

```bash
curl -X DELETE http://localhost:8080/models/mlp

{
  "status": "Workflow \"mlp\" unregistered"
}
```

### List workflows

`GET /models`

`GET /workflows`

* limit - (optional) the maximum number of items to return. It is passed as a query parameter. The default value is `100`.
* next_page_token - (optional) queries for next page. It is passed as a query parameter. This value is return by a previous API call.

Use the Workflows API to query current registered models and workflows:

```bash
curl "http://localhost:8080/workflows"
```

This API supports pagination:

```bash
curl "http://localhost:8080/models?limit=2&next_page_token=0"

{
  "models": [
    {
      "modelName": "mlp",
      "modelUrl": "https://resources.djl.ai/test-models/mlp.zip",
      "status": "READY"
    }
  ]
}
```
