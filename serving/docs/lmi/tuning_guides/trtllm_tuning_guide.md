# TensorRT LLM Tuning guide

This doc recommends the configurations based on your model and instance type.

The below table lists recommended TP and batch size parameters for a given model and instance.
The recommendations are for a **fixed input length of 1024 and max_new_tokens of 512.** If customers want to use different values of input and output lengths, they should gradually increase/decrease batch size or TP (if possible) appropriately.

| Model	          | Model size (GB)	 | Instance Type	 | Tensor Parallel Degree	 | Model size / total GPU memory	 | max_rolling_batch_size	 |
|-----------------|------------------|----------------|-------------------------|--------------------------------|-------------------------|
| Llama2 7B	      | 14	              | g5.2xlarge	    | 1	                      | 0.58333	                       | 16	                     |
| Llama2 13B	     | 26	              | g5.12xlarge	   | 4	                      | 0.27083	                       | 32	                     |
| Llama2 70B	     | 134	             | g5.48xlarge	   | 8	                      | 0.69792	                       | 8	                      |
| Llama2 70B	     | 134	             | p4d.24xlarge	  | 8	                      | 0.41875	                       | 128	                    |
| Falcon 7B	      | 16	              | g5.2xlarge	    | 1	                      | 0.66667	                       | 16	                     |
| Falcon 40B	     | 84	              | g5.48xlarge	   | 8	                      | 0.4375	                        | 32	                     |
| Falcon 40B	     | 84	              | p4d.24xlarge	  | 8	                      | 0.2625	                        | 128	                    |
| Code Llama 34B	 | 63	              | p4d.24xlarge	  | 8	                      | 0.19688	                       | 128	                    |

## Configurations

### Llama2 7B serving.properties

g5.2xlarge(needs to be compiled ahead of time on g5.12xlarge)

```
option.model_id={{s3url}}
option.tensor_parallel_degree=1
option.max_rolling_batch_size=16
option.max_input_len=1024
option.max_output_len=512
```

### Llama2 13B serving.properties

g5.12xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=4
option.max_rolling_batch_size=32
option.max_input_len=1024
option.max_output_len=512
```

### Llama2 70B serving.properties

g5.48xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=8
option.max_input_len=1024
option.max_output_len=512
```

p4d.24xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=128
option.max_input_len=1024
option.max_output_len=512
```

### Falcon 7B serving.properties

g5.2xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=1
option.max_rolling_batch_size=16
option.max_input_len=1024
option.max_output_len=512
```

### Falcon 40B serving.properties

g5.48xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=32
option.max_input_len=1024
option.max_output_len=512
```

p4d.24xlarge

```
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=128
option.dtype=fp16
option.max_input_len=1024
option.max_output_len=512
```

### Code Llama 34B serving.properties

```
option.model_id={{s3url}}
option.tensor_parallel_degree=8
option.max_rolling_batch_size=128
option.max_input_len=1024
option.max_output_len=512
```