# LMI Dist Tuning Guide

This doc recommends the configurations such as `tensor_parallel_degree`, `max_rolling_batch_size` and `max_rolling_batch_prefill_tokens` based on your model and instance type.
<br/>The table below outlines the recommended configurations for the most frequently requested models. These configurations have undergone testing with a `max_new_tokens` value of 256 and input of 8 tokens. The models were assessed using these parameters to ensure optimal performance.

| Model	        | Model size (GB)	 | Instance type	 | Tensor Parallel Degree	 | max_rolling_batch_size	 | max_rolling_batch_prefill_tokens	 |
|---------------|------------------|----------------|-------------------------|-------------------------|-----------------------------------|
| GPT-NeoX-20B	 | 39	              | g5.12xlarge	   | 4	                      | 64	                     | 19200	                            |
| Flan-T5 XXL	  | 42	              | g5.12xlarge	   | 4	                      | 64	                     | 	                                 |
| Flan-UL2	     | 37	              | g5.12xlarge	   | 4	                      | 64	                     | 	                                 |
| Llama2 7B	    | 13	              | g5.12xlarge	   | 1	                      | 32	                     | 11000	                            |
| Llama2-13B	   | 25	              | g5.12xlarge	   | 2	                      | 32	                     | 14500	                            |
| Llama2 70B	   | 129	             | g5.48xlarge	   | 8	                      | 8	                      | 	                                 |
| Llama2 70B	   | 129	             | p4d.24xlarge	  | 8	                      | 64	                     | 	                                 |
| Falcon 7b	    | 14	              | g5.2xlarge	    | 1	                      | 64	                     | 	                                 |
| Falcon 40b	   | 78	              | g5.48xlarge	   | 8	                      | 64	                     | 18000	                            |

## Configurations

`max_rolling_batch_prefill_tokens` is optional. If not set, DJLServing will allocate the maximum number of tokens that can fit in the memory. The empty values in the above table means, we did not set it explicitly in serving.properties, so DJLServing will set the default value based on available memory. 
Kindly check [this page](../configurations_large_model_inference_containers.md) for other configurations.

### GPT-NeoX-20B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=EleutherAI/gpt-neox-20b
option.tensor_parallel_degree=4
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
option.max_rolling_batch_prefill_tokens=19200
```

### Flan-T5 XXL serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=google/flan-t5-xxl
option.tensor_parallel_degree=4
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
```

### Flan-UL2 serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=google/flan-ul2
option.tensor_parallel_degree=4
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
```

### Llama2 7B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-7B-fp16
option.task=text-generation
option.tensor_parallel_degree=1
option.rolling_batch=auto
option.max_rolling_batch_size=32
option.max_rolling_batch_prefill_tokens=11000
```

### Llama2-13B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-13B-fp16
option.task=text-generation
option.tensor_parallel_degree=2
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=32
option.max_rolling_batch_prefill_tokens=14500
option.model_loading_timeout=7200
```

### Llama2 70B serving.properties

g5.48xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-70B-fp16
option.tensor_parallel_degree=8
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.model_loading_timeout=7200
```

p4d.24xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-70B-fp16
option.tensor_parallel_degree=8
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=64
option.model_loading_timeout=7200
```

### Falcon 7b serving.properties

g5.2xlarge

```
engine=MPI
option.model_id=tiiuae/falcon-7b
option.tensor_parallel_degree=1
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=64
```

### Falcon 40b serving.properties

g5.48xlarge

```
engine=MPI
option.model_id=tiiuae/falcon-40b
option.tensor_parallel_degree=8
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
option.max_rolling_batch_prefill_tokens=18000
option.model_loading_timeout=7200
```
