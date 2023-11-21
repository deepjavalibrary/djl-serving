# TransformersNeuronX tuning guide


The table below illustrates the recommended configurations for the most requested models. These models were tested using the configurations below, and were tested with `max_new_tokens` of 256, and input of eight tokens. 

| Model	             | Model size (GB)	 | Instance type	 | Tensor Parallel Degree	 | max_rolling_batch_size	 | n_positions	 |
|--------------------|------------------|----------------|-------------------------|-------------------------|--------------|
| Llama2 7b	         | 14	              | inf2.xlarge	   | 2	                      | 4	                      | 2048	        |
| Llama2 7b	         | 14	              | inf2.24xlarge	 | 8	                      | 4	                      | 2048	        |
| Llama2 13b (int8)	 | 13	              | inf2.xlarge	   | 2	                      | 4	                      | 512	         |
| Llama2 13b	        | 26	              | inf2.24xlarge	 | 8	                      | 4	                      | 2048	        |

**Current limitations:** Using the 2.15.0 release we are limited to `batch_size=4` as our max batch size when compiling using normal optimization flags. This can be increased to 8 by lowering the optimization level to 1 but this is a not a good solution. Currently, lower `tensor_parallel_degree` corresponds with significantly higher latency, so to improve latency by increasing the number of neuron cores that the model is split across. Both of these limitations are supposed to be addressed in the next release 2.16.0.


## Configurations

### Llama2-7b serving.properties

inf2.xlarge

```
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=llama-2-7b
option.batch_size=4
option.tensor_parallel_degree=2
option.n_positions=2048
option.rolling_batch=auto
option.dtype=fp16
option.model_loading_timeout=1600
```

inf2.24xlarge

```
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=llama-2-7b
option.batch_size=4
option.tensor_parallel_degree=8
option.n_positions=2048
option.rolling_batch=auto
option.dtype=fp16
option.model_loading_timeout=1600
```

### Llama2-13b serving.properties

inf2.xlarge - loading precompiled model

```
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=llama-2-13b-split
option.load_split_model=True
option.compiled_graph_path=llama-2-13b-compiled
option.batch_size=4
option.tensor_parallel_degree=2
option.load_in_8bit=true
option.n_positions=2048
option.rolling_batch=auto
option.dtype=fp16
option.model_loading_timeout=3600
```

inf2.24xlarge

```
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=llama-2-13b
option.batch_size=4
option.tensor_parallel_degree=2
option.load_in_8bit=true
option.n_positions=512
option.rolling_batch=auto
option.dtype=fp16
option.model_loading_timeout=3600
```

