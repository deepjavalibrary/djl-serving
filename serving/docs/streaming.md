# DJL Serving Streaming

When a model is run, typically the response is given all at once. However, there are also some cases that benefit from having the response given back as it is generated. For example, a LLM can return back the characters as they are generated. This allows for more dynamic UIs that seem speedier as user's could then read the output as they are written rather than waiting.

## Model Support

The first step to supporting streaming is to use a model that supports it.

For a model using Python, see the [Streaming Python configuration guide](streaming_config.md). This provides instructions for modifying your `handle()` function to use the streaming output. After the model is modified to support streaming, you must also add `option.enable_streaming=true` to the `serving.properties` to enable the streaming support.

For a Java model, this means that it must have a Block implementing [`StreamingBlock`](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/streaming/StreamingBlock.html) and a Translator implementing async [`StreamingTranslator`](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/streaming/StreamingTranslator.html). Right now, it is available through the DJL api but is not yet in DJL Serving.

## HTTP Streaming

This simplest way to support streaming is with [HTTP Chunked Encoding](https://en.wikipedia.org/wiki/Chunked_transfer_encoding). This allows HTTP/1 to send back data is chunks rather than all at once. You can access support for this through whatever HTTP API you are using to make the request. As an example, you can see how it is handled by the [JavaScript fetch Streams API](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Using_readable_streams).

## Pagination Streaming

It is also possible to implement streaming using pagination. The main use case for pagination is when HTTP streaming is not available. For example, if you have a proxy before the model server then the proxy must also support streaming. For those that do not such as Amazon SageMaker (currently), pagination will enable streaming.

The support for pagination is built on top of the [DJL Serving Caching Support](cache.md). It works by having the first request run asynchronously and stream the results to the cache. It will then provide a token to access the results. Subsequent requests using the token will return the elements from the cache. See the diagram below:

```
// Request 1 streams to the cache and returns the access token
Request 1 [input data] -------> Cache
    Token <-

// Following requests returns the currently available data from the cache
Request 2+ [token, start]
    Partial Output  <----------- Cache
```

For good results, it is important to properly configure the cache. The available configuration options can be found on the [cache page](cache.md). Keep in mind that if you use a horizontally scaling service with DJL Serving such as Amazon Elastic Container Service, Amazon SageMaker, or Kubernetes, they must share the same cache or persist requests from the same user to the same instance of DJL Serving. This means you must enable usage of one of the external cache variants like Amazon DynamoDB or Amazon S3 to share the same cache.

### Initial Request

To run a request with pagination, pass the query header `x-synchronous: false`. Once this is done, the response will include the header `x-next-token`. Note that when using this as part of SageMaker, you will need to use [X-Amzn-SageMaker-Custom-Attributes](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html#API_runtime_InvokeEndpoint_ResponseSyntax). In that case, it would be `X-Amzn-SageMaker-Custom-Attributes: x-synchronous=false` for the request and the response is also shown below.

```sh
curl -v -X POST "http://localhost:8080/invocations" \
    -H "content-type: application/json" \
    -H "x-synchronous: false" \
    -d "..."
```

```
> POST /invocations HTTP/1.1
> Host: localhost:8080
> User-Agent: curl/7.68.0
> Accept: */*
> content-type: application/json
> x-synchronous: false
> Content-Length: 73
>
* upload completely sent off: 73 out of 73 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< X-Amzn-SageMaker-Custom-Attributes: x-next-token=df87d942-7f39-48c8-a9af-d4e762a2ab1d
< x-next-token: df87d942-7f39-48c8-a9af-d4e762a2ab1d
< x-request-id: 0e805b7d-f1b6-49af-a595-d8dc35aee338
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 0
< connection: keep-alive
<
* Connection #0 to host localhost left intact
```

### Data Requests

After the initial request to queue the job, you can use the following requests to access the data. This is done by passing the `x-starting-token` as a header. Here are examples both directly and for SageMaker:

```sh
curl -v -X POST "http://127.0.0.1:8080/invocations" \
    -H "x-starting-token: df87d942-7f39-48c8-a9af-d4e762a2ab1d"
```

```sh
curl -v -X POST "http://127.0.0.1:8080/invocations" \
    -H "X-Amzn-SageMaker-Custom-Attributes: x-starting-token=df87d942-7f39-48c8-a9af-d4e762a2ab1d"
```

Within the response, it will include the streamed data concatenated together for everything that is computed at the time. It may also return the response header `x-next-token` that can be used to retrieve the following page of computed results. If the header `x-next-token` is not found in the response, this indicates that computation has finished and the last of the data was sent in that response.

Depending on the size of the results, it may be difficult to get all of the results at once. In that case, pass the `x-max-items` header to place a limit on the number of items streamed back at once. It the maximum number of items is not available at the time of the request, it will send as many as have already been computed. Note that these items are the number of "streamed element" from the model, not bytes. The number of bytes per item depends on the model and depending on the model can vary.
