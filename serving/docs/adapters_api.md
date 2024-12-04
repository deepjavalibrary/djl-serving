# DJL Serving Adapters Management API

**Note that this API is experimental and is subject to change.

DJL Serving provides a set of API allow user to manage adapters at runtime:

1. [Register an adapter](#register-an-adapter)
2. [Update an adapter](#update-an-adapter)
3. [Describe an adapter's status](#describe-adapter)
4. [Unregister an adapter](#unregister-an-adapter)
5. [List registered adapters](#list-adapters)

This is an extension of the [Management API](management_api.md) and can be accessed the same.

## Adapter Management APIs

### Register an adapter

`POST /models/{model_name}/adapters`

* name: The adapter name.
* src: The adapter src. It currently requires a file, but eventually an id or URL can be supported depending on the model handler.
* preload (optional): Whether to preload the adapter during initialization, defaults to `true`.
* pin (optional): Whether to pin the adapter, defaults to `false`. If this option is enabled, adapter will be preloaded, and the adapter is pinned during initialization. This helps certain latency sensitive adapters to be present in GPU memory without being evicted.
* All additional arguments will be treated as additional model-specific options and will be passed to the model during adapter registration

```bash
curl -X POST "http://localhost:8080/models/adaptecho/adapters?name=a1&src=/opt/ml/model/adapters/a1"

{
  "status": "Adapter a1 registered"
}
```

### Update an adapter

`POST /models/{model_name}/adapters/{adapter_name}/update`

* preload (optional): Whether to preload the adapter during initialization.
* pin (optional): Whether to pin the adapter. LoRA adapters can be pinned in GPU without being evicted from LRUCache. This helps certain latency sensitive adapters to be present in GPU memory without being evicted.
* All additional arguments will be treated as additional model-specific options and will be passed to the model during adapter registration

```bash
curl -X POST "http://localhost:8080/models/adaptecho/adapters/a1/update?pin=true"

{
  "status": "Adapter a1 updated"
}
```

### Describe adapter

`GET /models/{model_name}/adapters/{adapter_name}`

Use the Describe Adapter API to get the status of an adapter:

```bash
curl http://localhost:8080/models/adaptecho/adapters/a1

[
  {
    "name": "a1",
    "src": "/opt/ml/model/adapters/a1",
    "pin": false
  }
]
```

### Unregister an adapter

`DELETE /models/{model_name}/adapters/{adapter_name}`

Use the Unregister Adapter API to free up system resources:

```bash
curl -X DELETE http://localhost:8080/models/adaptecho/adapters/a1

{
  "status": "Adapter a1 unregistered"
}
```

### List adapters

`GET /models/{model_name}/adapters`

* limit (optional): the maximum number of items to return. It is passed as a query parameter. The default value is `100`.
* next_page_token (optional): queries for next page. It is passed as a query parameter. This value is return by a previous API call.

Use the Adapters API to query current registered adapters:

```bash
curl "http://localhost:8080/models/adaptecho/adapters"
```

This API supports pagination:

```bash
curl "http://localhost:8080/models/adaptecho/adapters?limit=5&next_page_token=0"

{
  "adapters": [
    {
      "name": "a1",
      "src": "/opt/ml/model/adapters/a1",
      "pin": false
    }
  ]
}
```

### Advanced

For the single model use case, the `/models/{model_name}` API prefix can be omitted resulting in queries such as `GET /adapters`.