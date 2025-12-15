# LMCache Performance Benefits for LMI Customers

LMCache is a KV cache offloading solution that dramatically improves inference performance for large language models serving workloads with repeated context. By offloading KV cache from GPU memory to CPU RAM or NVMe storage, LMCache enables efficient handling of long-context scenarios while delivering substantial latency improvements. LMCache support has been integrated into LMI since v17.

## Key Performance Improvements

Based on extensive testing across model sizes and context lengths, LMCache delivers **exceptional performance gains**:

* **CPU offloading**: Up to **28x speedup in TTFT** (achieved with Qwen 2.5-7B at 2M token context length)
* **NVMe-based offloading**: Up to **16x speedup in TTFT** (achieved with Qwen 2.5-72B at 1M token context length using O_DIRECT)

Our specific benchmarking on Qwen 8B (serving 460K tokens across 46 documents) demonstrates:

* **Time to First Token (TTFT)**: Reduced from 1.161s to 0.438s with CPU offloading (**2.65x faster**)
* **Total Request Latency**: Reduced from 52.978s to 24.274s (**2.18x faster**)

## Cache Backend Comparison

### CPU RAM Offloading (Recommended for Latency-Critical Workloads)

* **Best Performance**: Up to **28x** speedup in TTFT (at 2M tokens with Qwen 2.5-7B), fastest query TTFT (0.437s in benchmark)
* **Use Case**: Latency-sensitive applications requiring immediate response
* **Limitation**: Constrained by instance RAM capacity (e.g., 1.1TB on p4de.24xlarge)

### NVMe Storage with O_DIRECT

* **Strong Performance**: Up to **16x** speedup in TTFT (at 1M tokens with Qwen 2.5-72B), query TTFT of 0.731s (approaching CPU performance)
* **Massive Capacity**: Supports TB-scale caching for extensive document collections
* **Use Case**: Large-scale deployments with substantial context requirements
* **Configuration**: Enable `use_odirect: True` for optimal performance

## When to Use LMCache

The value of LMCache depends on your model size, context length requirements and GPU memory:

For a p4de.24xlarge -

|Model Size	|Context Length	|Recommendation	|
|---	|---	|---	|
|~1B	|< 2M	|No offloading needed	|
|	|2M – 30M	|Consider CPU offloading	|
|	|> 25M	|Consider Disk offloading	|
|~7-10B	|< 250K	|No offloading needed	|
|	|250K – 8M	|Consider CPU offloading	|
|	|> 6M	|Consider Disk offloading	|
|~70B+	|< 250K	|No offloading needed	|
|	|250K – 3M	|Consider CPU offloading	|
|	|> 2.5M	|Consider Disk offloading	|

**Key Insight**: Larger models benefit from LMCache at shorter context lengths because they consume more memory per token. A 72B model requires offloading around 500K tokens, while a 1.5B model only needs it beyond 2.5M tokens.

## How to use LMCache

ENV Variables should be set as shown here

```
OPTION_LMCACHE_CONFIG_FILE=/opt/ml/model/lmcache_config.yaml
OPTION_KV_TRANSFER_CONFIG={"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}
```

lmcache_config.yaml changes as per the backend. But if any of the backends are not configured correctly, vLLM defaults to CPU offloading. vLLM does not currently support specifying advanced LMCache configuration as ENV variables.

### On-Host RAM offloading

```
# lmcache_config.yaml
# 256 Tokens per KV Chunk
**chunk_size**: 256
# 5GB of Pinned CPU memory
**max_local_cpu_size**: 5.0 # Changes with Model Size
```

### On-Host NVME offloading

```
*# lmcache_config.yaml*
*# 256 Tokens per KV Chunk*
**chunk_size**: 256
*# Enable Disk backend*
**local_disk**: "file://tmp/cache/" # Fixed for SM customers
*# 5GB of Disk memory*
**max_local_disk_size**: 5.0 # Changes with Model Size
*# Disable OS page cache in favor of CPU Pinned Memory*
**extra_config**: {'use_odirect': True} # Fixed for SM customers
# 5GB of Pinned CPU memory
**max_local_cpu_size**: 5.0 # default
```

## Deployment Recommendations

1. **Configure CPU offloading** when instance RAM permits—it delivers optimal performance
2. **Use NVMe with O_DIRECT enabled** for workloads requiring larger cache capacity
3. **Implement session-based sticky routing** on SageMaker Classic to maximize cache hit rates
4. **Consider model architecture**: Models with different KV head configurations (e.g., Llama 3 8B vs Qwen 2.5-7B) will have different offloading thresholds

## Performance Validation

LMI container with LMCache demonstrates **performance parity** with open-source vLLM LMCache, ensuring enterprise customers receive the same optimization benefits with production-grade support and integration.
