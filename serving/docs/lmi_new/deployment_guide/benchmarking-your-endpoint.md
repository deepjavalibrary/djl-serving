# Benchmarking your Endpoint

Benchmarking your endpoint is crucial to understanding how the configuration and backend you are using handle the expected traffic.
This document will focus on how to benchmark your sagemaker endpoint with a lightweight tool, `awscurl`.

Within LMI, we use [`awscurl`](https://github.com/deepjavalibrary/djl-serving/tree/master/awscurl) for benchmarking.
`awscurl` is a tool that provides a `curl` like API to make requests to AWS Services. It provides the following features:

* Launch concurrent clients 
* Specify number of requests per client
* Specify a tokenizer to get accurate token counts
* Accepts a list of request bodies to benchmark your endpoint across a variety of input/output lengths, and generation parameters
* Reports metrics like:
  * Total Time taken for benchmark
  * Non 200 responses and error rate
  * Average, P50, P90, P99 latency
  * Requests Per second (TPS)
  * Tokens per Request (generated tokens)
  * Tokens per Second (total Throughput of system across all clients/requests)
  * Time to first Byte (for streaming, represents the time to first token)

We will now walk through some examples of using `awscurl` to benchmark your SageMaker Endpoint.

## awscurl Usage

To install awscurl, and setup AWS credentials, see the `awscurl` [documentation](https://github.com/deepjavalibrary/djl-serving/tree/master/awscurl).

To see a list of available options, run `./awscurl -h`.

The options we use in the examples below are:

* `-c`: number of concurrent clients
* `-N`: number of requests per client (c * N is the total number of requests made during the benchmark)
* `-n`: name of aws service to call (will always be `sagemaker` for this use-case)
* `-X`: HTTP method to use (will always be `POST` for this use-case)
* `--connect-timeout`: time in seconds to wait for response (sagemaker has default 60 second invocation time, so typically this is 60)
* `-d`: request body JSON (this is a single request body that will be used across all clients and requests)
* `--dataset`: a path to directory of files, or path to a single file that contain payloads 
* `-P`: print output in JSON
* `-t`: report tokens per second in benchmark metrics
* `-H`: custom HTTP Headers
* `-o`: output file prefix to save results of requests (1 file is generated per client)

Additionally, we recommend that you set the `TOKENIZER` environment variable to the value of your model's tokenizer (either HuggingFace Hub model id, or local path where the `tokenizer.json` file is present).
If a tokenizer is not specified, and tokens per second is requested, `awscurl` will use the number of words in the response to calculate token level metrics.

All the following examples use a sample sagemaker endpoint URL. You should replace this with your own endpoint URL.

### Usage with a single common payload

To run a benchmark with 10 concurrent clients, each issuing 30 requests, with a single common payload, we can run:

```shell
TOKENIZER=<tokenizer_id> ./awscurl -c 10 -N 30 -X POST \
  -n sagemaker https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/lmi-model/invocations \
  --connect-timeout 60 \
  -d '{"inputs":"The new movie that got Oscar this year","parameters":{"max_new_tokens":256, "do_sample":true}}' \
  -H 'Content-Type: application/json' \
  -P -t -o output.txt
```

After running the above command, you should see output like this (these are sample numbers, not guarantees of performance):
```shell
{
  "tokenizer": "TheBloke/Llama-2-13B-fp16",
  "totalTimeMills": 136154.71,
  "totalRequests": 300,
  "failedRequests": 0,
  "errorRate": 0.0,
  "concurrentClients": 10,
  "tps": 0.92,
  "tokenThroughput": 235.56,
  "totalTokens": 32000,
  "tokenPerRequest": 256,
  "averageLatency": 5433.92,
  "p50Latency": 5385.51,
  "p90Latency": 5412.18,
  "p99Latency": 6580.67,
  "timeToFirstByte": 5411.33
}
```

Additionally, you will find `output.txt.<c>` files that contain the responses per client (there will be c files in total).

### Usage with multiple payloads

You can also use `awscurl` to send multiple different payloads instead of a single payload.
To do so, you must provide the `--dataset` argument.

The dataset argument must point to a directory containing multiple files, with each file containing a single line with a request payload.
Alternatively, the dataset argument can point to a single file which contains line separated payloads.

As an example, we can construct a dataset directory with sample payloads like this:

```shell
mkdir prompts
echo '{"inputs":"The new movie that got Oscar this year","parameters":{"max_new_tokens":256, "do_sample":true}}' > prompts/prompt1.txt
echo '{"inputs":"\nwrite a program to add two numbers in python\n","parameters":{"max_new_tokens":256, "do_sample":true}}' > prompts/prompt2.txt
```

We can then use the multiple payloads like this:

```shell
TOKENIZER=<tokenizer_id> ./awscurl -c 10 -N 30 -X POST -n sagemaker https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/lmi-model/invocations \
  --connect-timeout 60 \
  --dataset prompts \
  -H 'Content-Type: application/json' \
  -P -t -o output.txt
```

### Usage with SageMaker Streaming Response

You can also use `awscurl` to invoke your endpoint with streaming responses. 
This assumes that you have configured your deployment configuration with `option.output_formatter=jsonlines` so that LMI streams responses.

To benchmark your endpoint with streaming, you must call the `/invocations-response-stream` API.

```shell
TOKENIZER=<tokenizer_id> ./awscurl -c 10 -N 30 -X POST -n sagemaker https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/lmi-model/invocations-response-stream \
  --connect-timeout 60 \
  --dataset prompts \
  -H 'Content-Type: application/json' \
  -P -t -o output.txt
```

With streaming, the `timeToFirstByte` metric is more meaningful. 
It represents the average latency across all requests between the request being issued and the first token being returned.


Previous: [Deploying your endpoint](deploying-your-endpoint.md)