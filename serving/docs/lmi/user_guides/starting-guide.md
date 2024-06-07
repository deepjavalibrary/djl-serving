# LMI Starting Guide

Starting with v0.27.0, the only required configuration for LMI containers is the `HF_MODEL_ID` environment variable.
LMI will apply optimizations and configurations based on the model architecture and available hardware, removing the need manually set them. 

Based on the selected container, LMI will automatically:

* select the best backend based on the model architecture 
* enable continuous batching if supported for the model architecture to increase throughput
* configure the engine and operation mode
* maximize hardware use through tensor parallelism
* calculate maximum possible tokens and size the KV-Cache
* enable CUDA kernels and optimizations based on the available hardware and drivers

The following code example demonstrates this configuration UX using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

This example will use the [TheBloke/Llama2-7b-fp16](https://huggingface.co/TheBloke/Llama-2-7B-fp16) model. 

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Setup role and sagemaker session
iam_role = sagemaker.get_execution_role() 
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

# Fetch the uri of the LMI container that supports vLLM, LMI-Dist, HuggingFace Accelerate backends
lmi_image_uri = image_uris.retrieve(framework="djl-deepspeed", version="0.27.0", region=region)

# Create the SageMaker Model object. In this example we let LMI configure the deployment settings based on the model architecture  
model = Model(
  image_uri=lmi_image_uri,
  role=iam_role,
  env={
    "HF_MODEL_ID": "TheBloke/Llama-2-7B-fp16",
  }
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
endpoint_name = sagemaker.utils.name_from_base("llama-7b-endpoint")
model.deploy(instance_type="ml.g5.2xlarge", initial_instance_count=1, endpoint_name=endpoint_name)
predictor = Predictor(
  endpoint_name=endpoint_name,
  sagemaker_session=sagemaker_session,
  serializer=JSONSerializer(),
  deserializer=JSONDeserializer(),
)

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

The following text-generation models are supported for optimized inference with the simplified configuration UX:

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct`, etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen2-beta-7B`, `Qwen/Qwen2-beta-7B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)
- T5 (`google/flan-t5-xxl`, `google/flan-t5-base`, etc.)

Other text-generation models, and non-text generation models are also supported, but may not be as performant as the model architectures listed above.

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
