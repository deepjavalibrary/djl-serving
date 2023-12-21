# LMI Dist Tuning Guide


This doc recommends the configurations such as `tensor_parallel_degree` based on your model and instance type.

The table below outlines the recommended configurations for the most frequently requested models. These configurations have undergone testing with a `max_new_tokens` value of 256 and input of 8 tokens. The models were assessed using these parameters to ensure optimal performance.

|Model	|Model size (GB)	|Instance type	|Tensor Parallel Degree	|max_rolling_batch_size	|
|---	|---	|---	|---	|---	|
|GPT-NeoX-20B	|39	|g5.12xlarge	|4	|64	|
|Flan-T5 XXL	|42	|g5.12xlarge	|4	|64	|
|Flan-UL2	|37	|g5.12xlarge	|4	|64	|
|MPT 30B	|65	|g5.12xlarge	|4	|64	|
|Falcon 7B	|14	|g5.2xlarge	|1	|64	|
|Falcon 40B	|78	|g5.48xlarge	|8	|64	|
|Llama2 7B	|13	|g5.2xlarge	|1	|32	|
|Llama2 13B	|25	|g5.12xlarge	|2	|32	|
|Llama2 70B	|129	|g5.48xlarge	|8	|8	|
|Llama2 70B	|129	|p4d.24xlarge	|8	|64	|
|Llama2 13B GPTQ	|7	|g5.2xlarge	|1	|64	|
|Llama2 70B GPTQ	|35	|g5.12xlarge	|4	|64	|
|CodeLlama 34B	|63	|g5.12xlarge	|4	|64	|
|Mistral-7B-v0.1	|15	|g5.12xlarge	|4	|64	|

## Configurations

Kindly check [this page](http://../configurations_large_model_inference_containers.md) for configurations.

### GPT-NeoX-20B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=EleutherAI/gpt-neox-20b
option.tensor_parallel_degree=4
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
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

### MPT 30B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=mosaicml/mpt-30b
option.tensor_parallel_degree=4
option.rolling_batch=lmi-dist
option.trust_remote_code=true
option.dtype=fp16
option.max_rolling_batch_size=64
option.model_loading_timeout=7200
```

### Falcon 7B serving.properties

g5.2xlarge

```
engine=MPI
option.model_id=tiiuae/falcon-7b
option.tensor_parallel_degree=1
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=64
```

### Falcon 40B serving.properties

g5.48xlarge

```
engine=MPI
option.model_id=tiiuae/falcon-40b
option.tensor_parallel_degree=8
option.rolling_batch=auto
option.dtype=fp16
option.max_rolling_batch_size=64
option.model_loading_timeout=7200
```

### Llama2 7B serving.properties

g5.2xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-7B-fp16
option.task=text-generation
option.tensor_parallel_degree=1
option.rolling_batch=auto
option.max_rolling_batch_size=32
```

### Llama2 13B serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-13B-fp16
option.task=text-generation
option.tensor_parallel_degree=2
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=32
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

### Llama2 13B GPTQ serving.properties

g5.2xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-13B-GPTQ
option.tensor_parallel_degree=1
option.max_rolling_batch_size=64
option.rolling_batch=auto
option.quantize=gptq
```

### Llama2 70B GPTQ serving.properties

g5.12xlarge

```
engine=MPI
option.model_id=TheBloke/Llama-2-70B-GPTQ
option.tensor_parallel_degree=4
option.max_rolling_batch_size=64
option.rolling_batch=auto
option.model_loading_timeout=7200
option.quantize=gptq
```

### CodeLlama 34B serving.properties

```
engine=MPI
option.model_id=codellama/CodeLlama-34b-hf
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=64
option.model_loading_timeout=7200
```

### Mistral-7B-v0.1 serving.properties

```
engine=MPI
option.model_id=mistralai/Mistral-7B-v0.1
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=lmi-dist
option.max_rolling_batch_size=8
```
