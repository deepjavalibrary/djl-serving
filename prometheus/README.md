# DJLServing prometheus metrics support

## DJLServing metrics

DJL provide the types of metrics:

- Server metrics: saved in `$MODELSERVER_HOME/logs/server_metrics.log`
- Model metrics: save in `$MODELSERVER_HOME/logs/model_metric.log`

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
| RESPONSE_4_XX     | Number of requests failed with 4XX response                    | Count        | Server metric      |
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
option.log_request_metric=true
```

By default, per request metrics will be aggregated every `1000` requests. You can change
aggregated with the following:

```
option.metrics_aggregation=100
```
