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

* url - Workflow url
* template - A workflow template to use
* engine - the name of engine to load the model. DJL will try to infer engine if not specified.
* device - the device to load the model. DJL will pick optimal device if not specified, the value device can be:
    * CPU device: cpu or simply -1
    * GPU device: gpu0, gpu1, ... or simply 0, 1, 2, 3, ...
* min_worker - the minimum number of worker processes. The default value is `1`.
* max_worker - the maximum number of worker processes. The default is the same as the setting for `min_worker`.
* synchronous - if the creation of worker is synchronous. The default value is true.

Either a url or [template](workflow_templates.md) is required.

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

### Configure server logging

`POST /server/logging`

* level - set logging level: TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF, if no log level specified, `INFO` level will be set.

```bash
curl -X POST "http://localhost:8080/server/logging?level=debug"

{
  "status": "OK"
}
```

### Get metrics

`GET /server/metrics`

`GET /metrics`

* name[] - metric name filters. Multiple `name[]` HTTP parameters are allowed

Use the get metrics to get metrics in prometheus text format. The metrics come from two sources, and
each one has its own switch:

| Environment variable      | Metrics                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `SERVING_PROMETHEUS`      | The metrics of DJL Serving itself, see the [metrics list](../../prometheus/README.md).        |
| `SERVING_ENGINE_METRICS`  | The metrics that an inference engine records inside the Python workers, for example vLLM's.   |

Both are off unless you set them to `true`. The `lmi` container image sets `SERVING_ENGINE_METRICS` to `true`.

| `SERVING_ENGINE_METRICS` | `SERVING_PROMETHEUS` | `GET /server/metrics`                           | `GET /metrics` |
|--------------------------|----------------------|-------------------------------------------------|----------------|
| not `true`               | not `true`           | `400`                                           | `404`          |
| not `true`               | `true`               | `200`, the metrics of DJL Serving               | `404`          |
| `true`                   | not `true`           | `200`, the engine metrics                       | same as `/server/metrics` |
| `true`                   | `true`               | `200`, the metrics of DJL Serving, then the engine metrics | same as `/server/metrics` |

On `/server/metrics`, any method other than `GET` gets `405`. On `/metrics` the same holds when
`SERVING_ENGINE_METRICS` is `true`; otherwise `/metrics` is `404` for every method. `/metrics` is
served on the management address, like `/server/metrics`. If `management_address` differs from
`inference_address`, both paths are only on the management address.

```
curl "http://localhost:8080/server/metrics?name[]=DJLServingStart&name[]=StartupLatency"

# HELP DJLServingStart_total : prometheus counter metric, unit: Count
# TYPE DJLServingStart_total counter
DJLServingStart_total{Version="0.27.0-SNAPSHOT"} 1.0
# HELP StartupLatency : prometheus gauge metric, unit: Microseconds
# TYPE StartupLatency gauge
StartupLatency{Host="localhost"} 425941.0
```

#### Engine metrics

With `SERVING_ENGINE_METRICS=true`, each Python worker serves the metric families whose name starts
with `vllm:` on a Unix domain socket in the temporary directory, and DJL Serving fetches them when
you call `/server/metrics` or `/metrics`. Nothing listens on a new network port.

```
curl "http://localhost:8080/metrics?name[]=vllm:num_requests_running"

# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="facebook/opt-125m"} 0.0
```

* The answer is always the Prometheus text format, whatever the `Accept` header says.
* The answer is sent when the workers have answered, so a client has to keep its side of the
  connection open until it has read the answer. A client that closes its sending side right after the
  request (for example `nc -N`) gets no answer. `/ping` behaves the same.
* The workers export only the `vllm:` families of their `prometheus_client` registry. Metrics that
  your own `model.py` registers, and the `python_*` and `process_*` families, are not exported.
* The `model_name` label is the served model name: `option.served_model_name` when it is set (the
  example above sets it to `facebook/opt-125m`), and otherwise the path of the model files in the
  container.
* `name[]` also selects engine families. A name matches a family by its name, by its name plus
  `_total`, or by the name of one of its samples, for example `vllm:request_success_total` or
  `vllm:e2e_request_latency_seconds_bucket`.
* One worker: the text of the worker as it is. More than one worker (for example
  `tensor_parallel_degree` below the number of GPUs): the families are merged, and every sample gets
  a `worker` label with the device of its worker (`gpu0`, `gpu1`, or `gpu0-1` for a second worker on
  the same device). With `SAGEMAKER_MULTI_MODEL=true` every sample has the label, even with one
  worker. Without it the label appears when a second worker starts and goes away when only one is
  left, so the series change their labels at that moment. A worker that restarts keeps its label,
  and its counters start again at zero.
* A worker that does not answer within 2 seconds, or whose answer is larger than 4 MiB, is left out
  and a warning is logged, at most once every five minutes for each worker. If workers are
  registered and none of them answers, the answer is `503`.
* An identical request (the same `name[]` filters) is answered from the previous fetch for one
  second, so scraping the same way more often does not add load on the workers. A request with
  other filters starts its own fetch, and only the most recent one is kept. At most 32 fetches can be
  unfinished at once. A request that would start another one gets `503` at once, so a burst of
  requests with many different filters does not queue up behind a slow worker. A warning is logged
  for that, at most once every five minutes.
* The answer follows the worker process. If the engine process inside a running worker dies, the
  worker still answers, with the values it recorded last, for as long as the worker process lives.
* The workers need `prometheus_client`, which vLLM installs. A worker without it serves nothing.
  Workers that run under MPI (for example TensorRT-LLM), on several nodes, or without Unix domain
  sockets (Windows) do not serve engine metrics, and the answer has no engine families.
* The setting is read when the server and its workers start.
