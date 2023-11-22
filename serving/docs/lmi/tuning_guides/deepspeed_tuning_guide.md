# DeepSpeed tuning guide

The table below illustrates the recommended configurations for the most requested models. These models were tested using the configurations below, and were tested with `max_new_tokens` of 256, and input of eight tokens. 

| Model	                  | Model size (GB)	 | Instance type	 | Tensor Parallel Degree	 | max_rolling_batch_size	 |
|-------------------------|------------------|----------------|-------------------------|-------------------------|
| Llama2 7B	              | 14	              | g5.2xlarge	    | 1	                      | 8	                      |
| Llama2 7B smoothquant	  | 7	               | g5.2xlarge	    | 1	                      | 16	                     |
| Llama2 7B	              | 14	              | g5.12xlarge	   | 2	                      | 32	                     |
| Llama2 13B	             | 26	              | g5.12xlarge	   | 4	                      | 32	                     |
| Llama2 13B smoothquant	 | 13	              | g5.12xlarge	   | 4	                      | 64	                     |

## Configurations

### Llama2 7B serving.properties

g5.2xlarge

```
engine=DeepSpeed
option.model_id=s3://djl-sm-test/llama-2-7b-hf/
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.tensor_parallel_degree=1
```

```
engine=DeepSpeed
option.model_id=s3://djl-sm-test/llama-2-7b-hf/
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=16
option.tensor_parallel_degree=1
option.quantize=smoothquant
```

g5.12xlarge

```
engine=DeepSpeed
option.model_id=s3://djl-sm-test/llama-2-7b-hf/
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=32
option.tensor_parallel_degree=2
```

### Llama2 13B serving.properties

g5.12xlarge

```
engine=DeepSpeed
option.model_id=s3://djl-sm-test/llama-2-13b-hf/
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=32
option.tensor_parallel_degree=4
```

```
engine=DeepSpeed
option.model_id=s3://djl-sm-test/llama-2-13b-hf/
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=64
option.tensor_parallel_degree=4
option.quantize=smoothquant
```
