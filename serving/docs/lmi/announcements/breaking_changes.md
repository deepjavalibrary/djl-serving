# LMI Breaking Changes

This document details breaking changes for LMI container version releases.

## LMI Container v12 (0.30.0)

### Changes to model loading/warmup behavior.

#### Summary
Starting in LMI v12 (0.30.0), the default behavior for `option.max_rolling_batch_prefill_tokens` when using the lmi-dist 
rolling-batch backend has changed.
This configuration maps to vLLM's `max_num_batched_tokens` configuration, and will be updated to match this name in a future version.
In previous versions (v11 and earlier), this value would default to 4096 tokens.
Starting in v12, the default behavior for lmi-dist will rely on vLLM's default behavior.
vLLM's default behavior is described [here](https://github.com/vllm-project/vllm/blob/9cc373f39036af789fb1ffc1e06b23766996d3f4/vllm/config.py#L959C9-L988).
The previous behavior was implemented to improve the default experience for model loading, 
especially on memory-constrained instances. 
vLLM's default behavior is to set this value to `option.max_model_len`, and use this as part of the warmup phase. 
After the warmup phase, lmi-dist would set this value back to `option.max_model_len`.
This allowed users to deploy models with long context lengths on smaller instances without needing
to configure `option.max_rolling_batch_prefill_tokens` or `option.max_model_len`. 
However, if users attempted requests with longer than 4096 tokens of prefill, 
there was a chance this could lead to out-of-memory scenarios.

#### What to expect
We have decided to make this change because of chunked prefill, our observation that more users are leveraging longer prompts, 
and our belief that the previous motivation is no longer the correct default experience.
This change means that if `option.max_rolling_batch_prefill_tokens` is not specified, some deployments that worked previously
with v11 and earlier may start failing in v12 due to higher memory requirements during model loading and warmup 
for models with greater than a supported sequence length greater than 4096 tokens.

In v12, the default behavior is to allocate enough memory to store the KV-cache for `option.max_model_len` tokens.
For models with large context lengths (or any context length above 4096 tokens), more memory will be required by default
during model load and startup time. As a result, some previous configurations when deployed with v12 will require
additional memory by default, which can lead to failures at startup.

#### Guidance
As a result of this change, we recommend you take the following actions:

* Enable chunked-prefill. You can do this by setting `option.enable_chunked_prefill=true`. 
  * You can learn more about chunked prefill here https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill.
  * We expect that chunked prefill will be beneficial for most users and use-cases. But, you should confirm this for your specific use-case.
  * With chunked-prefill, the default value for `option.max_rolling_batch_prefill_tokens` is 512, which is set to optimize inter token latency (ITL).
  * Starting in LMI v12 (0.30.0), the default value for `option.enable_chunked_prefill` is `None` rather than `False`. This brings LMI default
    in line with OSS vllm default, which results in chunked prefill being enabled by default for models with a context length greater than 32k.
  * The default logic in vLLM is described [here](https://github.com/vllm-project/vllm/blob/9cc373f39036af789fb1ffc1e06b23766996d3f4/vllm/config.py#L959C9-L988).
* Tune `option.max_rolling_batch_prefill_tokens` and `option.max_model_len` based on available accelerator memory (instance type), use-case, and model. 
  * `option.max_model_len` should be set to the maximum input + output token count you expect to support at inference time.
    * If you are unable to use a certain `option.max_model_len` for a model on an instance, you will need to reduce this value, use an instance type with more gpu memory, or explore techniques like quantization to reduce the memory required for the model.
  * `option.max_rolling_batch_prefill_tokens` depends on whether you use chunked prefill.
    * With chunked prefill, you can typically expect higher throughput with larger values, and better latency with lower values.
    * Without chunked prefill, you should set this to the maximum input length you aim to support.

## LMI-TensorrtLLM v12 (0.30.0)

### T5 model family memory increase 

We have observed slightly different behavior for the t5 model family compared to our previous v11 release (0.29.0).
We have raised an issue with Tensorrt-LLM [here](https://github.com/NVIDIA/TensorRT-LLM/issues/2398). 
The change in behavior is likely due to how Tensorrt-LLM handles memory for the kv-cache.

#### Guidance
As a result, we recommend that you carefully tune the batch size and sequence length for your model.
Configurations that previously worked on v11 may lead to Out-Of-Memory scenarios with v12.

### T5 model family batch scheduler policy

We have noticed runtime issues when using the default `batch_scheduler_policy` of `max_utilization` for t5 models.
When this is combined with in-flight batching (the default), it leads to inference errors.

#### Guidance
We recommend users that are deploying t5 model variants to specify `option.batch_scheduler_policy=guaranteed_no_evict`
to ensure proper inference execution.
