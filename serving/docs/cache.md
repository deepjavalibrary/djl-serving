# DJL Serving Async and Caching

DJL Serving has support for asynchronous request and request caching. The asynchronous requests can be used when the model is large and may cause timeouts waiting for it to compute the response. In this case, an asynchronous request will complete immediately without timeout concerns and later requests can be used to get the result and/or check if it is completed. The cache choice is global and can be set using environment variables or system properties (see below).

It is also possible to use this as an LRU cache to avoid recomputing common inputs. To enable this, apply the multi-tenant cache configuration (see below). This use case is currently experimental.

### Initial Request

To run an asynchronous request, pass the query header `x-synchronous: false`. Once this is done, the response will include the header `x-next-token`. Note that when using this as part of SageMaker, you will need to use [X-Amzn-SageMaker-Custom-Attributes](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html#API_runtime_InvokeEndpoint_ResponseSyntax). In that case, it would be `X-Amzn-SageMaker-Custom-Attributes: x-synchronous=false` for the request and the response is also shown below.

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

If the result is not yet available, you will receive a response with HTTP code `202`.

## Cache Configuration

In addition, there are a number of options that can be used to configure the cache. In DJL, cache support is enabled by an implementation of the `CacheEngine`. The first option to choose is which `CacheEngine` to use. The `MemoryCacheEngine` is the default and the only one that is always available. The other cache engines require the use of the [DJL Serving Cache Plugin](http://docs.djl.ai/docs/serving/plugins/cache/index.html).

Keep in mind that if you use a horizontally scaling service with DJL Serving such as Amazon Elastic Container Service, Amazon SageMaker, or Kubernetes, they must share the same cache. This means you must enable usage of one of the external cache variants like Amazon DynamoDB or Amazon S3.

There are also several properties (both environment variables and system properties) that apply to all engines:

`SERVING_CACHE_MULTITENANT` (default false) indicates to use a multi-tenant cache with keys based on the hash of the input. Using this, two users with the same input would share the same cache entry, allowing the later one to avoid recomputing it. The alternative has a UUID for each cache entry ensuring they are unique. This feature is still experimental.

`SERVING_CACHE_BATCH` (default 1) indicates how many streaming items to store in the cache at once. It can be used to reduce writes for granular streaming like text generation. For example, setting it to 5 with DDB would mean each DB save would include 5 characters instead of 1.

### Memory Cache

The memory cache is the default cache that stores data in memory on the machine (JVM). It is not suitable for use cases with horizontal scaling. It supports the following properties (environment variables and system properties):

`SERVING_MEMORY_CACHE_CAPACITY` (default none) provides an optional maximum capacity for the cache. This makes it follow an LRU strategy.

### DDB Cache

The DDB cache is based on [Amazon DynamoDB](https://aws.amazon.com/dynamodb/). This cache requires the cache plugin. It can be used for horizontal scaling and is recommended for smaller outputs like text. It supports the following properties (environment variables and system properties):

`SERVING_DDB_CACHE` can be set to "true" to use the DDB cache.

`SERVING_DDB_TABLE_NAME` (default "djl-serving-pagination-table") sets the table to use for the cache.

There are a few final notes. The default `SERVING_CACHE_BATCH` when using the DDB cache is 5. It also does not support multi-tenant.

### S3 Cache

The S3 cache is based on [Amazon S3](https://aws.amazon.com/s3/). This cache requires the cache plugin. It can be used for horizontal scaling and is recommended for larger outputs like images. It supports the following properties (environment variables and system properties):

`SERVING_S3_CACHE` can be set to "true" to use the S3 cache.

`SERVING_S3_CACHE_BUCKET` (required) sets the name of the bucket to use.

`SERVING_S3_CACHE_KEY_PREFIX` (default "") sets a prefix for the caching path in the bucket. For example, a prefix of "serving/cache/" with an entry "xxx" would make the entry have the combined path "serving/cache/xxx". It can be used to reuse a bucket or share a bucket with other use cases.

`SERVING_S3_CACHE_AUTOCREATE` (default false) can be set to "true" to automatically create the S3 bucket if it does not exist.
