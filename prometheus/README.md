# DJLServing prometheus metrics support

## DJLServing metrics

DJL provide the types of metrics:

- Server metrics: saved in `$MODEL_SERVER_HOME/logs/server_metric.log`
- Model metrics: save in `$MODEL_SERVER_HOME/logs/model_metric.log`

## Prometheus metrics

Those metrics can be collected as prometheus metrics, you need set `SERVING_PROMETHEUS` environment
variable or System properties to enable it.

Once you enabled prometheus metrics, you can use [get metric management REST API](../serving/docs/management_api.md#get-metrics) to get metrics.


## Built-in Metrics


| Metric name       | Description                                                    | Unit         | Type               |
|-------------------|----------------------------------------------------------------|--------------|--------------------|
| DJLServingStart   | Counter for server startup                                     | Count        | Server metric      |
| StartupLatency    | Server startup latency                                         | Microseconds | Server metric      |
| GPUMemory_*       | GPU memory                                                     | Bytes        | Server metric      |
| Response_2XX      | Number of requests succeeded with 2XX response                 | Count        | Server metric      |
| Response_4XX      | Number of requests failed with 4XX response                    | Count        | Server metric      |
| Response_5XX      | Number of requests failed with 5XX response                    | Count        | Server metric      |
| ServerError       | Number of requests failed due to unknown exception             | Count        | Server metric      |
| WlmError          | Number of requests failed due to exceed job queue limit        | Count        | Server metric      |
| CmdError          | Counter for server startup failure due to invalid command line | Count        | Server metric      |
| StartupFailed     | Counter for server startup failure due to unknown exception    | Count        | Server metric      |
| ModelLoadingError | Counter for server startup failure due to failed to load model | Count        | Server metric      |
| RegisterWorkflow  | Load workflow (model) latency                                  | Microseconds | Per model metric   |
| QueueTime         | Time waiting in the queue                                      | Microseconds | Per model metric   |
| DynamicBatchSize  | Dynamic batch size                                             | Count/Item   | Per model metric   |
| RequestLatency    | Request latency                                                | Microseconds | Per model metric   |
| DownloadModel     | Download model latency                                         | Microseconds | Per model metric   |
| LoadModel         | Load model latency                                             | Microseconds | Per model metric   |
| LoadAdapter       | Load PEFT Adapter latency                                      | Microseconds | Per model metric   |
| ConvertTrtllm     | Convert TensorRT-LLM model latency                             | Microseconds | Per model metric   |
| RollingBatchSize  | Average rolling batch size                                     | Count/Item   | Per request metric |
| TokenLatency      | Average token latency                                          | Microseconds | Per request metric |
| TokenThroughput   | Average per token throughput                                   | Count/Second | Per request metric |
| OutputTokens      | Average output tokens per request                              | Count/Item   | Per request metric |
| Preprocess        | Average pre-processing latency                                 | Microseconds | Per request metric |
| Inference         | Average model inference call latency                           | Microseconds | Per request metric |
| Postprocess       | Average post-processing latency                                | Microseconds | Per request metric |
| Prediction        | Average model prediction call latency                          | Microseconds | Per request metric |

### Per request metric

Per request metrics is disabled by default, set the following option to enable per request metric:

```
log_request_metric=true
```

By default, per request metrics will be aggregated every `1000` requests. You can change
aggregated with the following:

```
metrics_aggregation=100
```

## Inference engine metrics

An inference engine that runs inside a Python worker, for example vLLM in the LMI container,
records its own Prometheus metrics in the worker process. DJL Serving can serve them next to its
own. Set the `SERVING_ENGINE_METRICS` environment variable to `true` (the `lmi` container image
does) and call `GET /server/metrics` or `GET /metrics`, see the
[get metrics REST API](../serving/docs/management_api.md#get-metrics). The engine metrics do not
need `SERVING_PROMETHEUS`. When both are `true`, the answer holds the DJL Serving metrics first and
the engine metrics after them.

```
curl "http://localhost:8080/metrics?name[]=vllm:kv_cache_usage_perc"

# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="facebook/opt-125m"} 0.0
```

Each worker serves the families whose name starts with `vllm:`, with the names, types, help texts
and label keys that vLLM uses. The `model_name` label is the value of `option.served_model_name`
(the example sets it to `facebook/opt-125m`), and the path of the model files in the container when
that option is not set. The latency clocks start when a request reaches the engine in the
worker, so the time that a request waits in DJL Serving before that is not in them.
With more than one worker, or with `SAGEMAKER_MULTI_MODEL=true`, every sample has an extra `worker`
label. Without `SAGEMAKER_MULTI_MODEL=true` the label appears when a second worker starts and goes
away when only one is left, so the series change their labels at that moment.
