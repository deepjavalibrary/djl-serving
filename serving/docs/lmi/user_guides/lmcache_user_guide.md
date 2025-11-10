# LMCache User Guide

## Overview

LMCache is a KV cache sharing mechanism that allows multiple requests with shared prefixes to reuse cached computations, significantly improving performance and reducing latency for scenarios with common context.

## Configuration

### Required serving.properties

Add the following to your `serving.properties`:

```properties
option.kv_transfer_config={"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}
```

### Environment Variable

Set the LMCache configuration file path:

```bash
LMCACHE_CONFIG_FILE=/opt/ml/model/test/lmcache_config.yaml
```

## Configuration Options

### 1. CPU Backend (lmcache_cpu.yaml)

```yaml
chunk_size: 256
local_device: cpu
remote_url: null
```

**Use case**: Store KV cache in CPU memory for sharing across requests.

### 2. Local Storage Backend (lmcache_local_storage.yaml)

```yaml
chunk_size: 256
local_device: cpu
remote_url: file:///tmp/lmcache
```

**Use case**: Persist KV cache to local disk storage for sharing across server restarts.

## Example Usage

### serving.properties
```properties
engine=Python
option.model_id=s3://djl-llm/llama-3-8b-instruct-hf/
option.tensor_parallel_degree=4
option.kv_transfer_config={"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}
```

## Performance Benefits

- **Reduced TTFT**: Time to first byte can be reduced for requests with shared prefixes
- **Higher Throughput**: More requests can be processed with the same resources for common prefix scenarios

**Note**: In testing, no significant performance benefit was observed in typical workloads.

## Caveats

- **Redis connector**: Currently not working
- **S3 connector**: Currently not working
- Only CPU and local storage backends are functional

## Requirements

- `kv_role` must be set to `kv_both` in the kv_transfer_config
- LMCache configuration file must be accessible at the specified path
