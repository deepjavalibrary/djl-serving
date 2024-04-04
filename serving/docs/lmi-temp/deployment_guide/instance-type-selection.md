# Instance Type Selection

While there are many open source LLMs and architectures available, most models tend to fall within a few common parameter count sizes.
The following table provides instance type recommendations for common model parameter counts using half-precision (fp16/fp32) weights.

| Model Parameter Count | Instance Type    | Accelerators | Aggregate Accelerator Memory | Sample Models                              | Estimated Max Batch Size Range |
|-----------------------|------------------|--------------|------------------------------|--------------------------------------------|--------------------------------|
| ~7 billion            | ml.g5.2xlarge    | 1 x A10G     | 24 GB                        | Llama2-7b, Falcon-7b, GPT-J-6B, Mistral-7b | 32-64                          |
| ~13 billion           | ml.g5.12xlarge   | 4 x A10G     | 96GB                         | Llama2-13b, CodeLlama-13b, Flan-T5-XXL     | 32-64                          |
| ~20 billion           | ml.g5.12xlarge   | 4 x A10G     | 96GB                         | GPT-NEOX-20b, Flan-Ul2                     | 16-32                          |
| ~35 billion           | ml.g5.48xlarge   | 8 x A10G     | 192GB                        | CodeLlama-34b, Falcon-40b                  | 32-64                          |
| ~70 billion           | ml.g5.48xlarge   | 8 x A10G     | 192GB                        | Llama2-70b, CodeLlama-70b                  | 1-8                            |
| ~70 billion           | ml.p4d.24xlarge  | 8 x A100     | 320GB                        | Llama2-70b, CodeLlama-70b                  | 32-64                          |
| ~180 billion          | ml.p4de.24xlarge | 8 x A100     | 640GB                        | Falcon-180b, Bloom-176B                    | 32-64                          |

We recommend starting with the guidance above based on the model parameter count.
The estimated batch size is a conservative estimate.
You will likely be able to increase the batch size beyond the recommendation, but that is dependent on your model and expected max sequence lengths (prompt + generation tokens).

For a more in-depth instance type sizing guide, you can follow the steps below.

Selecting an instance is based on a few factors:

* Model Size
* Desired Accelerators (A10, A100, H100, AWS Inferentia etc)
    * We recommend using instances with at least A series gpus (g5/p4). The performance is much greater compared to older T series gpus
    * You should select an instance that has sufficient aggregate memory (across all gpus) for both loading the model, and making requests at runtime
* Desired Concurrency/Batch Size
    *  Increasing Batch Size allows for more concurrent requests, but requires additional VRAM

We will walk through a sizing example using the Llama2-13b model.

## Model Size

We can establish a lower bound for the required memory based on the model size.
The model size is determined by the number of parameters, and the data type.
We can quickly estimate the model size in GB using the number of parameters and data type like this:

* Half Precision data type (fp16, bf16): `Size in GB = Number of Parameters * 2 (bytes / param)`
* Full Precision data type (fp32): `Size in GB = Number of Parameters * 4 (bytes / param)`
* 8-bit Quantized data type (int8): `Size in GB = Number of Parameters * 1 (bytes / param)`

We recommend using a half precision data type as it requires less memory than full precision without losing accuracy for most cases.

We estimate the Llama2-13b model to take `13 billion params * 2 bytes / param = 26GB` of memory.
This is just the memory required to load the model.
To execute inference, additional memory is required at runtime.

## Additional Runtime Memory

To estimate the additional memory required at runtime, we need to estimate the size of the KV cache.
The KV cache can be thought of as the state of your model for a given generation loop.
It stores the Key (K) and Value (V) states for each attention layer in your model.

To estimate the size of the KV cache, we will use the following formula.

`KV-Cache Size (bytes / token) = 2 * n_dtype * n_layers * n_hidden_size`

Breaking down this formula:

* 2 comes from the two matrices we need to cache: Key (K), and Value (V)
* n_dtype represents the number of bytes per parameter, which is based on the data type (4 for fp32, 2 for fp16)
* n_layers represents the number of transformer blocks in the model (usually `num_hidden_layers` in model's `config.json`)
* n_hidden_size represents the dimension of the attention block (num_heads * d_head matrix) (usually `hidden_size` in model's `config.json`)

For the Llama2-13b model, we get the kv-cache size per token as:

* `2 * 2 * 5120 * 40 = 819,200 bytes/token = ~0.00082 GB / token`

For a single max sequence (4096 tokens for the Llama2-13b model), we require `4096 token * 0.00082 GB / token = 3.36 GB`

## Selecting an Instance Type

Now that we know the memory required to load the model, and have an estimate of the runtime memory required per token, we can figure out what instance type to use.
We recommend that you have an understanding of the max sequence length you will be operating with (prompt tokens + generation tokens).
Alternatively, you can select an instance type and calculate and max batch size estimate based on the available memory.

Please see this [link](https://aws.amazon.com/sagemaker/pricing/) for an up-to-date list of available instance types on SageMaker.

For our Llama2-13b model, we need a minimum of 26GB.
Let's consider two instances types: ml.g5.12xlarge, and ml.g5.48xlarge 

On the ml.g5.12xlarge instance, we will have `96GB - 26GB = 66GB` available for batches of sequences at runtime.
This would provide us enough memory for roughly 85,300 tokens at a time. This means we can have:

* batch size of ~20 for maximum sequence length of 4096 tokens
* batch size of ~40 for maximum sequence length of 2048 tokens
* And so on

On a ml.g5.48xlarge, we will have `192GB - 26GB = 166GB` available for batches of sequences at runtime
This would provide us enough memory for roughly 202,400 tokens at runtime. This means we can have:

* batch size of ~49 for maximum sequence length of 4096
* batch size of ~98 for maximum sequence length of 2048
* And so on

This exercise demonstrates how to estimate the memory requirements of your model and use case in order to select an instance type.
The calculations made are estimates, but should serve as a good starting point for testing. 
Memory allocation behavior and memory optimization features differ between backends, so actual runtime memory usage will be different than what was derived above.
We recommend testing your expected traffic against your setup for a specific instance type to determine the proper configuration.

Next: [Backend Selection](backend-selection.md)
