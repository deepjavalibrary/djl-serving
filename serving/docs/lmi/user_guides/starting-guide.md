# LMI Starting Guide

Starting with v0.27.0, the only required configuration for LMI containers is the `HF_MODEL_ID` environment variable.
LMI will apply optimizations and configurations based on the model architecture and available hardware, removing the need to manually set them. 

Based on the selected container, LMI will automatically:

* select the best backend based on the model architecture 
* enable continuous batching if supported for the model architecture to increase throughput
* configure the engine and operation mode
* maximize hardware use through tensor parallelism
* calculate maximum possible tokens and allocate the KV-Cache
* enable CUDA kernels and optimizations based on the available hardware and drivers

The following code example demonstrates this configuration UX using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

This example will use the [TheBloke/Llama2-7b-fp16](https://huggingface.co/TheBloke/Llama-2-7B-fp16) model. 

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker.djl_inference import DJLModel 

# Setup role and sagemaker session
iam_role = sagemaker.get_execution_role() 
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

# Create the SageMaker Model object. In this example we let LMI configure the deployment settings based on the model architecture  
model = DJLModel(
  model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
  role=iam_role,
  env={
    "HF_TOKEN": "<hf token value for gated models>",
  }
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
endpoint_name = sagemaker.utils.name_from_base("llama-8b-endpoint")
predictor = model.deploy(instance_type="ml.g5.12xlarge", initial_instance_count=1, endpoint_name=endpoint_name)

# Make an inference request against the llama2-7b endpoint
outputs = predictor.predict({
  "inputs": "The diamondback terrapin was the first reptile to be",
  "parameters": {
    "do_sample": True,
    "max_new_tokens": 256,
  }
})
```

## Supported Model Architectures

The following models are supported for optimized inference with the simplified configuration UX:

Text Generation Models

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Arctic (`Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- Command-R (`CohereForAI/c4ai-command-r-v01`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct` etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon & Falcon2 (`tiiuae/falcon-7b`, `tiiuae/falcon-11b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- Gemma2 (`google/gemma-2-9b`, `google/gemma-2-27b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- InternLM2 (`internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.)
- Jais (`core42/jais-13b`, `core42/jais-13b-chat`, `core42/jais-30b-v3`, `core42/jais-30b-chat-v3`, etc.)
- Jamba (`ai21labs/Jamba-v0.1`, etc.)
- LLaMA, Llama 2, Llama 3, Llama 3.1 (`meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc.)
- MiniCPM (`openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OLMo (`allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Orion (`OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Phi-3 (`microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen1.5-7B`, `Qwen/Qwen1.5-7B-Chat`, etc.)
- Qwen2MoE (`Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.)
- StableLM(`stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.)
- Starcoder2(`bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.)
- T5 (`google/flan-t5-xxl`, `google/flan-t5-base`, etc.)
- Xverse (`xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.)

Multi Modal Models

- Chameleon (`facebook/chameleon-7b` etc.)
- Fuyu (`adept/fuyu-8b` etc.)
- LlaVA-1.5 (`llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-1.5-13b-hf`, etc.)
- LlaVA-NeXT (`llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc.)
- PaliGemma (`google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, etc.)
- Phi-3-Vision (`microsoft/Phi-3-vision-128k-instruct`, etc.)

For models not specified above, please check the specific inference engine user-guides to see if they are supported.

## Available Environment Variable Configurations

The following environment variables are exposed as part of this simplified UX:

**HF_MODEL_ID**

This configuration is used to specify the location of your model artifacts.
It can either be a HuggingFace Hub model-id (e.g. TheBloke/Llama-2-7B-fp16), a S3 uri (e.g. s3://my-bucket/my-model/), or a local path.
If you are using [SageMaker's capability to specify uncompressed model artifacts](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-uncompressed.html), you should set this value to `/opt/ml/model`.
`/opt/ml/model` is the path in the container where model artifacts are mounted if using this mechanism.

**HF_REVISION**

If you are using a model from the HuggingFace Hub, this specifies the commit or branch to use when downloading the model.

This is an optional config, and does not have a default value. 

**HF_TOKEN**

Some models on the HuggingFace Hub are gated and require permission from the owner to access.
To deploy a gated model from the HuggingFace Hub using LMI, you must provide an Access Token via this environment variable.

**HF_MODEL_TRUST_REMOTE_CODE**

If the model artifacts contain custom modeling code, you should set this to true after validating the custom code is not malicious.
If you are using a HuggingFace Hub model id, you should also specify `HF_REVISION` to ensure you are using artifacts and code that you have validated.

This is an optional config, and defaults to `False`.

**TENSOR_PARALLEL_DEGREE**

This value is used to specify the number of GPUs to partition the model across using tensor parallelism.
The value should be less than or equal to the number of available GPUs on the instance type you are deploying with.
We recommend setting this to `max`, which will shard the model across all available GPUs.

This is an optional config, and defaults to `max`.

**MAX_BATCH_SIZE**

This value represents the maximum number of requests the model will handle in a batch.

This is an optional config, and defaults to `256`.

**MAX_CONCURRENT_REQUESTS**

This value represents the number of active requests/client connections the model server can handle.
This value should be greater than or equal to `MAX_BATCH_SIZE`. 
For requests received when a full batch is being executed, they will be queued until a free slot in the batch becomes available.

This is an optional config, and defaults to `1000`.

### Additional Configurations

Additional configurations are available to further tune and customize your deployment.
These configurations are covered as part of the advanced deployment guide [here](../deployment_guide/configurations.md).
In most cases, the simplified configuration set described here is sufficient to achieve high performance.
We recommend you first deploy using the configurations described here, and venture into the advanced configurations if you need additional customizations and performance tuning.

The configurations described in that doc follow a different naming convention when using the `serving.properties` configuration format.
The above environment variables translate to the following `serving.properties` configurations:

* `HF_MODEL_ID`: `option.model_id`
* `HF_REVISION`: `option.revision`
* `HF_MODEL_TRUST_REMOTE_CODE`: `option.trust_remote_code`
* `TENSOR_PARALLEL_DEGREE`: `option.tensor_parallel_degree`
* `MAX_BATCH_SIZE`: `option.max_rolling_batch_size`
* `MAX_CONCURRENT_REQUESTS`: `job_queue_size`

## API Schema

The request and response schema for interacting with the model endpoint is available [here](lmi_input_output_schema.md).
