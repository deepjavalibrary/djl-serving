# TensorRT-LLM(TRT-LLM) Engine User Guide

## Model Artifacts Structure 

TRT-LLM LMI supports two options for model artifacts

1.  [Standard HuggingFace model format](../deployment_guide/model-artifacts.md#huggingface-transformers-pretrained-format): In this case, TRT-LLM LMI will build TRT-LLM engines from HuggingFace model and package them with HuggingFace model config files during model load time.
2. [Custom TRT-LLM LMI model format](../deployment_guide/model-artifacts.md#tensorrt-llmtrt-llm-lmi-model-format): In this case, artifacts are directly loaded without the need to model compilation resulting in faster load times.



## Supported Model Architectures

The below model architectures are supported for JIT model compiltation and tested in our CI.

* LLaMA (since LMI V7 0.25.0)
* Falcon (since LMI V7 0.25.0)
* InternLM (since LMI V8 0.26.0)
* Baichuan (since LMI V8 0.26.0)
* ChatGLM (since LMI V8 0.26.0)
* GPT-J (since LMI V8 0.26.0)
* Mistral (since LMI V8 0.26.0)
* Mixtral (since LMI V8 0.26.0)
* Qwen (since LMI V8 0.26.0)
* GPT2/SantaCoder (since LMI V8 0.26.0)

TRT-LLM LMI v8 0.26.0 containers come with [TRT-LLM 0.7.1](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.7.1). For models that are not listed here and supported by [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.7.1?tab=readme-ov-file#models) with [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend), you can use this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_manual_convert_tutorial.md) instead to prepare model manually.

We will add more model support in the future versions in our CI. Please feel free to [file an issue](https://github.com/deepjavalibrary/djl-serving/issues/new/choose) if you are looking for a specific model support.


## SageMaker Deployment Tutorial
Users need to provide the model id of the model they want to deploy. Model id can be provided using `OPTION_MODEL_ID` environment variable which can take one of the following values
* Hugging Face model id
* s3 uri of Hugging Face model stored in s3
* s3 uri of pre-compiled TRT-LLM LMI model artifacts

We also need to set `SERVING_LOAD_MODELS` environment variable which can be set as below. 

```
SERVING_LOAD_MODELS=test::MPI=/opt/ml/model
OPTION_MODEL_ID=<your model id>
```

We also support customizing configuration to boost performance according to your specific use case. Please refer to `Common` and `TensorRT-LLM` sections in this [doc](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/configurations_large_model_inference_containers.md)

In this tutorial, we will use [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) to deploy the model on SageMaker. The below code can be run in SageMaker environment to deploy llama2-13b on g5.12xlarge instance. 

```
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Setup role and sagemaker session
iam_role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

# Fetch the uri of the TRT-LLM LMI container
container_image_uri = image_uris.retrieve(framework="djl-tensorrtllm", version="0.26.0", region=region)

# Create the SageMaker Model object. In this example we'll use vLLM as our inference backend
model = Model(
  image_uri=container_image_uri,
  role=iam_role,
  env={
    "SERVING_LOAD_MODELS": "model_name::MPI=/opt/ml/model",
    "OPTION_MODEL_ID": "TheBloke/Llama-2-13B-fp16",
  }
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
endpoint_name = sagemaker.utils.name_from_base("llama-13b-trtllm-endpoint")
model.deploy(instance_type="ml.g5.12xlarge", initial_instance_count=1, endpoint_name=endpoint_name)
predictor = Predictor(
  endpoint_name=endpoint_name,
  sagemaker_session=sagemaker_session,
  serializer=JSONSerializer(),
  deserializer=JSONDeserializer(),
)


# Make an inference request against the llama2-13b endpoint
outputs = predictor.predict({
  "inputs": "The diamondback terrapin was the first reptile to be",
  "parameters": {
    "do_sample": True,
    "max_new_tokens": 256,
  }
})
print(outputs)
```


##  Advanced tuning guide



| Item | LMI Version | Required | Description | Example value |
|------|----------|----------|-----------|------------------|
| option.max_input_len	| >= 0.25.0 | No	      | Maximum input token size you expect the model to have per request. This is a compilation parameter that set to the model for Just-in-Time compilation. If you set this value too low, the model will unable to consume the long input. | Default values for:<br/>Llama is `512` <br/> Falcon is `1024`	                                                                                                              |
| option.max_output_len	          |   >= 0.25.0           | No	      | Maximum output token size you expect the model to have per request. This is a compilation parameter that set to the model for Just-in-Time compilation. If you set this value too low, the model will unable to produce tokens beyond the value you set.	                                                                                                                                                                                                                                                                                                                                            | Default values for:<br/> Llama is `512` <br/> Falcon is `1024`	                                                                                                             |
| option.use_custom_all_reduce	    |     >= 0.25.0        | No	      | Custom all reduce kernel is used for GPUs that have NVLink enabled. This can help to speed up model inference speed with better communication. Turn this on by setting true on P4D, P4De, P5 and other GPUs that are NVLink connected	                                                                                                                                                                                                                                                                                                                                                               | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| Advanced parameters	                          |
| option.tokens_per_block	          |  >= 0.25.0          | No	      | tokens per block to be used in paged attention algorithm	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Default values is `64`	                                                                                                                                                     |
| option.batch_scheduler_policy	        |  >= 0.25.0      | No	      | scheduler policy of Tensorrt-LLM batch manager.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `max_utilization`, `guaranteed_no_evict` <br/> Default value is `max_utilization`	                                                                                          |
| option.kv_cache_free_gpu_mem_fraction	   |  >= 0.25.0   | No	      | fraction of free gpu memory allocated for kv cache. The larger value you set, the more memory the model will try to take over on the GPU. The more memory preserved, the larger KV Cache size we can use and that means longer input+output sequence or larger batch size.	                                                                                                                                                                                                                                                                                                                          | float number between 0 and 1. <br/> Default is `0.95`	                                                                                                                      |
| option.max_num_sequences	       |     >= 0.25.0         | No	      | maximum number of input requests processed in the batch. We will apply max_rolling_batch_size as the value for it if you don't set this. Generally you don't have to touch it unless you really want the model to be compiled to a batch size that not the same as model server set	                                                                                                                                                                                                                                                                                                                 | Integer greater than 0 <br/> Default value is the batch size set while building Tensorrt engine	                                                                            |
| option.enable_trt_overlap	        |     >= 0.25.0       | No	      | Parameter to overlap the execution of batches of requests. It may have a negative impact on performance when the number of requests is too small. During our experiment, we saw more negative impact to turn this on than off.                                                                                                                                                                                                                                                                                                                                                                       | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.enable_kv_cache_reuse	    |   >= 0.26.0             | No	      | This feature is only supported for GPT-like model on TRTLLM (as of 0.7.1) and need to compile the model with `--use_paged_context_fmha`. Let the LLM model to remember the last used input KV cache and try to reuse it in the next run. An instant benefit will be blazing fast first token latency. This is typically helpful for document understanding, chat applications that usually have the same input prefix. The TRTLLM backends will remember the prefix tree of the input and reuse most of its part for the next generation. However, this does come with the cost of extra GPU memory. | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.baichuan_model_version	    |   >= 0.26.0          | No	      | Parameter that exclusively for Baichuan LLM model to specify the version of the model. Need to specify the HF Baichuan checkpoint path. For v1_13b, you should use whether baichuan-inc/Baichuan-13B-Chat or baichuan-inc/Baichuan-13B-Base. For v2_13b, you should use whether baichuan-inc/Baichuan2-13B-Chat or baichuan-inc/Baichuan2-13B-Base. More Baichuan models could be found on baichuan-inc.	                                                                                                                                                                                            | `v1_7b`, `v1_13b`, `v2_7b`, `v2_13b`. <br/> Default is `v1_13b`	                                                                                                            |
| option.chatglm_model_version      |     >= 0.26.0        | No             | Parameter exclusive to ChatGLM models to specify the exact model type. Required for ChatGLM models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `chatglm_6b`, `chatglm2_6b`, `chatglm2_6b_32k`, `chatglm3_6b`, `chatglm3_6b_base`, `chatglm3_6b_32k`, `glm_10b`. <br/> Default is `unspecified`, which will throw an error. |
| option.gpt_model_version       | >= 0.26.0            | No             | Parameter exclusive to GPT2 models to specify the exact model type. Required for GPT2 models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `gpt2`, `santacoder`, `starcoder`. <br/> Default is `gpt2`. |
| option.multi_block_mode     | >= 0.26.0              | No             | Split long kv sequence into multiple blocks (applied to generation MHA kernels). It is beneifical when `batch x num_heads` cannot fully utilize GPU. This is **not** supported for qwen model type.                                                                                                                                                                                                                                                                                                                                                                                                  | `true`, `false`. <br/> Default is `false`	 |
| option.use_fused_mlp   | >= 0.26.0                | No             | Enable horizontal fusion in GatedMLP, reduces layer input traffic and potentially improves performance for large Llama models(e.g. llama-2-70b). This option is only supported for Llama model type.                                                                                                                                                                                                                                                                                                                                                                                                 | `true`, `false`. <br/> Default is `false`	 |
| option.rotary_base    | >= 0.26.0               | No             | Rotary base parameter for RoPE embedding. This is supported for llama, internlm, qwen model types                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `float` value. <br/> Default is `10000.0`	 |
| option.rotary_dim |   >= 0.26.0              | No             | Rotary dimension parameter for RoPE embedding. This is supported for only gptj model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `int` value. <br/> Default is `64`	 |
| option.rotary_scaling_type </br> option.rotary_scaling_factor  |>= 0.26.0               | No             | Rotary scaling parameters. These two options should always be set together to prevent errors. These are supported for llama, qwen and internlm models                                                                                                                                                                                                                                                                                                                                                                                                                                                | The value of `rotary_scaling_type` can be either `linear` and `dynamic`. The value of `rotary_scaling_factor` can be any value larger than 1.0. Default is `None`.|
| Advanced parameters: SmoothQuant	 |
| option.quantize	    | >= 0.26.0                          | No	      | Currently only supports `smoothquant` for Llama, Mistral, InternLM and Baichuan models with just in time compilation mode.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `smoothquant`	                                                                                                                                                              |
| option.smoothquant_alpha	 | >= 0.26.0                     | No	      | smoothquant alpha parameter	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Default value is `0.8`	                                                                                                                                                     |
| option.smoothquant_per_token	| >= 0.26.0                  | No	      | This is only applied when `option.quantize` is set to `smoothquant`.  This enables choosing at run time a custom smoothquant scaling factor for each token. This is usally little slower and more accurate	                                                                                                                                                                                                                                                                                                                                                                                          | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.smoothquant_per_channel	| >= 0.26.0                | No	      | This is only applied when `option.quantize` is set to `smoothquant`.  This enables choosing at run time a custom smoothquant scaling factor for each channel. This is usally little slower and more accurate	                                                                                                                                                                                                                                                                                                                                                                                        | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.multi_query_mode	| >= 0.26.0                       | No	      | This is only needed when `option.quantize` is set to `smoothquant` . This is should be set for models that support multi-query-attention, for e.g llama-70b	                                                                                                                                                                                                                                                                                                                                                                                                                                         | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| Advanced parameters: AWQ	                     |
| option.quantize	    | >= 0.26.0                           | No	      | Currently only supports `awq` for Llama and Mistral models with just in time compilation mode.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `awq`	                                                                                                                                                                      |
| option.awq_format	   | >= 0.26.0                   | No	      | This is only applied when `option.quantize` is set to `awq`. awq format you want to set. Currently only support `int4_awq`                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Default value is `int4_awq`	                                                                                                                                                |
| option.awq_calib_size	 | >= 0.26.0                 | No	      | This is only applied when `option.quantize` is set to `awq`. Number of samples for calibration. 	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Default is `32`	                                                                                                                                                            |


