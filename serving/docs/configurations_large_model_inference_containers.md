# Large Model Inference Containers

There are a number of shared configurations for python models running large language models.
They are also available through the [Large Model Inference Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers).

### Common ([doc](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html))

| Item                    | Required | Description                                                                                                                                                                                                                                                                       | Example value                                                           |
|-------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| engine                  | Yes      | The runtime engine of the code                                                                                                                                                                                                                                                    | `Python, DeepSpeed, FasterTransformer, MPI`                             |
| option.model_dir        | No       | The directory path to load the model. Default is set to the current path with model files.                                                                                                                                                                                        | Default: `/opt/djl/ml`                                                  |
| option.model_id         | No       | The value of this option will be the Hugging Face ID of a model or the s3 url of the model artifacts. DJL Serving will use the ID to download the model from Hugging Face or the s3 url. DJL Serving uses `s5cmd` to download the model from the bucket which is generally faster | `google/flan-t5-xl`, `s3://<my-bucket>/google/flan-t5-xl` Default: None |
| option.enable_streaming | No       | Enables response streaming. Use `huggingface` to enable HuggingFace like streaming output                                                                                                                                                                                         | `false`, `true`, `huggingface`                                          |
| option.dtype            | No       | Datatype to which you plan to cast the model default.                                                                                                                                                                                                                             | `fp16, fp32, bf16, int8`                                                |

### DeepSpeed ([doc](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html))

| Item                      | Required | Description                                                                                                                                                                                     | Example value                  |
|---------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| option.task               | No       | The task used in Hugging Face for different pipelines.                                                                                                                                          | `text-generation`              |
| option.max_tokens         | No       | Total number of tokens (input and output) with which DeepSpeed can work. The number of output tokens in the difference between the total number of tokens and the number of input tokens.       | `1024`                         |
| option.low_cpu_mem_usage  | No       | Reduce CPU memory usage when loading models. We recommend that you set this to True.                                                                                                            | `true`                         |
| option.enable_cuda_graph  | No       | Activates capturing the CUDA graph of the forward pass to accelerate.                                                                                                                           | `true`                         |
| option.triangular_masking | No       | Whether to use triangular masking for the attention mask. This is application or model specific.                                                                                                | `false`                        |
| option.return_tuple       | No       | Whether transformer layers need to return a tuple or a tensor.                                                                                                                                  | `false`                        |
| option.training_mp_size   | No       | If the model was trained with DeepSpeed, this indicates the tensor parallelism degree with which the model was trained. Can be different than the tensor parallel degree desired for inference. | `2`                            |
| option.checkpoint         | No       | Path to DeepSpeed compatible checkpoint file.                                                                                                                                                   | `ds_inference_checkpoint.json` |
| option.parallel_loading   | No       | Loads multiple workers in parallel (faster but risks running out of memory).                                                                                                                    | `ds_inference_checkpoint.json` |

### FasterTransformer ([doc](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html))

[Common settings](#common-doc)

### HuggingFace ([doc](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html))

| Item                                    | Required | Description                                                                                                                                                                             | Example value                                            |
|-----------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| option.task                             | No       | The task used in Hugging Face for different pipelines.                                                                                                                                  | `text-generation`                                        |
| option.load_on_devices                  | No       | delimited devices list, which the model to be loaded on, default to load on all devices.                                                                                                | `0, 1`                                                   |
| option.trust_remote_code                | No       | Set to True to use a HF hub model with custom code                                                                                                                                      | Default: `false`                                         |
| option.load_in_4bit                     | No       | Use `bitsandbytes` quantization. Supported only on certain models.                                                                                                                      | Default: `false`                                         |
| option.load_in_8bit                     | No       | Use `bitsandbytes` quantization. Supported only on certain models.                                                                                                                      | Default: `false`                                         |
| option.low_cpu_mem_usage                | No       | Reduce CPU memory usage when loading models. We recommend that you set this to True.                                                                                                    | `true`                                                   |
| option.revision                         | No       | Use a particular version/commit hash of a HF hub model                                                                                                                                  | `ed94a7c6247d8aedce4647f00f20de6875b5b292` Default: None |
| option.rolling_batch                    | No       | Enable iteration level batching using one of the supported strategies                                                                                                                   | `auto, scheduler, lmi-dist, vllm, neuron, trtllm`        |
| option.quantize                         | No       | Only supported for `option.rolling_batch=lmi-dist`. Only supported value is `bitsandbytes`                                                                                              | `bitsandbytes` Default: `None`                           |
| option.paged_attention                  | No       | Only supported for `option.rolling_batch=lmi-dist`. Enabling this would require more GPU memory to be preallocated for caching                                                          | Default: `true`                                          |
| option.max_rolling_batch_prefill_tokens | No       | Only supported for `option.rolling_batch=lmi-dist`. Limits the number of tokens for caching. This needs to be tuned based on batch size and input sequence length to avoid GPU OOM      | Default: `1088`                                          |
| option.decoding_strategy                | No       | Only supported for `option.rolling_batch=scheduler` Specifies the decoding method among sample, greedy and contrastive search                                                           | `sample, greedy, contrastive` Default: `greedy`          |
| option.max_sparsity                     | No       | Only supported for `option.rolling_batch=scheduler` Used in max-sparsity thresholding mechanism. It limits the max_sparsity in the token sequence caused by padding.                    | `0.01, 0.5` Default: `0.33`                              |
| option.max_splits                       | No       | Only supported for `option.rolling_batch=scheduler` Used in max-sparsity thresholding mechanism. It limits the max number of batch splits, where each split has its own inference call. | `1, 5` Default: `3`                                      |
| option.max_rolling_batch_size           | No       | Limits the number of concurrent requests                                                                                                                                                | Default: `32`                                            |

### Transformers-NeuronX ([doc](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-configuration.html))

| Item               | Required | Description                                                                                                                            | Example value   |
|--------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| option.n_positions | No       | Input sequence length                                                                                                                  | Default: `128`  |
| option.unroll      | No       | Unroll the model graph for compilation. With `unroll=None` compiler will have more opportunities to do optimizations across the layers | Default: `None` |

## Aliases

DJLServing provides a few alias for Python engine to make it easy for common LLM configurations.

- `engine=DeepSpeed`, equivalent to:

```
engine=Python
option.mpi_mode=true
option.entryPoint=djl_python.deepspeed
```

- `engine=FasterTransformer`, this is equivalent to:

```
engine=Python
option.mpi_mode=true
option.entryPoint=djl_python.fastertransformer
```

- `engine=MPI`, this is equivalent to:

```
engine=Python
option.mpi_mode=true
option.entryPoint=djl_python.huggingface
```
