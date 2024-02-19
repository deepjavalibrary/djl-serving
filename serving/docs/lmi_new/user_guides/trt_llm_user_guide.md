# TRT-LLM 

## Model Artifacts Structure 

TensorRT-LLM(TRT-LLM) LMI supports two options for model artifacts

1.  Standard HuggingFace model format: In this case, TRT-LLM LMI will build TRT-LLM engines from HuggingFace model and package them with HuggingFace model config files during model load time.
2. Custom TRT-LLM LMI model format: In this case, users are expected to pre-compile TRT-LLM engines from HuggingFace model and package them with Hugging Face model config files. With this option, model loading time will be lesser than option 1. Users can create these artifacts for model architectures that are supported for JIT compilation following this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_aot_tutorial.md). For model architectures that are not supported by TRT-LLM LMI for JIT compilation, follow this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_manual_convert_tutorial.md) to create model artifacts.

 Below directory structure represents an example TensorRT-LLM LMI model artifacts structure.

```
  trt_llm_model_repo
  ├──
    └── tensorrt_llm
        ├── 1
        │ ├── trt_llm_model_float16_tp2_rank0.engine # trt-llm engine
        │ ├── trt_llm_model_float16_tp2_rank1.engine # trt-llm engine
        │ ├── config.json # trt-llm config file
        │ └── model.cache
        ├── config.pbtxt # trt-llm triton backend config
        ├── config.json # Below are HuggingFace model config files
        ├── pytorch_model.bin.index.json
        ├── requirements.txt
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.model
```



## Supported Model Architectures

The below model architectures are supported for JIT model compiltation

* LLaMA (since LMI V7 0.25.0)
* Falcon (since LMI V7 0.25.0)
* InternLM (since LMI V8 0.26.0)
* Baichuan (since LMI V8 0.26.0)
* ChatGLM (since LMI V8 0.26.0)
* GPT-J (since LMI V8 0.26.0)
* Mistral (since LMI V8 0.26.0)
* Mixtral (since LMI V8 0.26.0)
* Qwen (since LMI V8 0.26.0)
* GPT2/SantaCoder/StarCoder (since LMI V8 0.26.0)

For model that are not listed here and supported by TRT-LLM, you can use this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_manual_convert_tutorial.md) instead to prepare model manually.


## Tutorial to deploy with default configs for a single model on an endpoin

To deploy the model on SageMaker, users need to create a `serving.properties` configuration file that will be read by djl-serving. 
For TRT-LLM LMI, users need to specify `option.model_id` property which can take one of the values discussed in model artifacts section. Below are the valid options for `option.model_id`

* Hugging Face model id
* s3 uri of Hugging Face model stored in s3
* s3 uri of pre-compiled TRT-LLM LMI model artifacts


Refer to `Common` and `TensorRT-LLM` sections in this [doc](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/configurations_large_model_inference_containers.md) to set custom values for supported properties and fine tune model hosting.


### Create model artifacts

1. Create a `serving.properties` file

```
option.model_id=<your model>
option.tensor_parallel_degree=1
option.max_rolling_batch_size=16 # default is 32
option.max_input_len=256 # default is 1024
option.max_output_len=256 # default is 512
```

Users can also set the properties using environment variables. This [doc](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/lmi_environment_variable_instruction.md) explains how to convert serving.properties to environment variables.


1. Create a tar file with serving.properties

```
%%sh
mkdir model
mv serving.properties model/
tar czvf model.tar.gz model/
rm -rf model
```

### Deploy the model using the SageMaker SDK

1. Create a SageMaker session

```
import boto3
import sagemaker

role = sagemaker.get_execution_role()
sess = sagemaker.session.Session()
aws_region = sess._region_name
```

1. Create a model in SageMaker.

```
from sagemaker import Model, image_uris, serializers

# fetch TRT-LLM image uri
image_uri = image_uris.retrieve(
                framework="djl-tensorrtllm",
                region=sess.boto_session.region_name,
                version="0.26.0"
                )
                
# publish model artifacts to s3
s3_model_prefix = "your_s3_model_prefix"
bucket = sess.default_bucket()  # bucket to house artifacts
model_s3_url = sess.upload_data("mymodel.tar.gz", bucket, s3_model_prefix)
     

# create sagemaker model
model = Model(
    image_uri=image_uri,
    model_data=model_s3_url,
    role=role,
   )
```

1. Deploy model

```
instance_type = "<choose_your_instance_type>"
endpoint_name = sagemaker.utils.name_from_base("trt-llm-lmi-model")
model.deploy(initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    )

# our requests and responses will be in json format so we specify the serializer and the deserializer
predictor = sagemaker.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sess,
        serializer=serializers.JSONSerializer(),
        )
```



### Run inference

```
predictor.predict(
{"inputs": "tell me a story of the little red riding hood", "parameters": {"max_new_tokens":128, "do_sample":True}}
)
```



### Clean up environment

```
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
model.delete_model()
```



##  Advanced tuning guide [ Hidden unless clicked] 

| Item                                          | Required | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Example value                                                                                                                                                               |
|-----------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| option.max_input_len	                         | No	      | Maximum input token size you expect the model to have per request. This is a compilation parameter that set to the model for Just-in-Time compilation. If you set this value too low, the model will unable to consume the long input.	                                                                                                                                                                                                                                                                                                                                                              | Default values for:<br/>Llama is `512` <br/> Falcon is `1024`	                                                                                                              |
| option.max_output_len	                        | No	      | Maximum output token size you expect the model to have per request. This is a compilation parameter that set to the model for Just-in-Time compilation. If you set this value too low, the model will unable to produce tokens beyond the value you set.	                                                                                                                                                                                                                                                                                                                                            | Default values for:<br/> Llama is `512` <br/> Falcon is `1024`	                                                                                                             |
| option.use_custom_all_reduce	                 | No	      | Custom all reduce kernel is used for GPUs that have NVLink enabled. This can help to speed up model inference speed with better communication. Turn this on by setting true on P4D, P4De, P5 and other GPUs that are NVLink connected	                                                                                                                                                                                                                                                                                                                                                               | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| Advanced parameters	                          |
| option.tokens_per_block	                      | No	      | tokens per block to be used in paged attention algorithm	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Default values is `64`	                                                                                                                                                     |
| option.batch_scheduler_policy	                | No	      | scheduler policy of Tensorrt-LLM batch manager.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `max_utilization`, `guaranteed_no_evict` <br/> Default value is `max_utilization`	                                                                                          |
| option.kv_cache_free_gpu_mem_fraction	        | No	      | fraction of free gpu memory allocated for kv cache. The larger value you set, the more memory the model will try to take over on the GPU. The more memory preserved, the larger KV Cache size we can use and that means longer input+output sequence or larger batch size.	                                                                                                                                                                                                                                                                                                                          | float number between 0 and 1. <br/> Default is `0.95`	                                                                                                                      |
| option.max_num_sequences	                     | No	      | maximum number of input requests processed in the batch. We will apply max_rolling_batch_size as the value for it if you don't set this. Generally you don't have to touch it unless you really want the model to be compiled to a batch size that not the same as model server set	                                                                                                                                                                                                                                                                                                                 | Integer greater than 0 <br/> Default value is the batch size set while building Tensorrt engine	                                                                            |
| option.enable_trt_overlap	                    | No	      | Parameter to overlap the execution of batches of requests. It may have a negative impact on performance when the number of requests is too small. During our experiment, we saw more negative impact to turn this on than off.                                                                                                                                                                                                                                                                                                                                                                       | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.enable_kv_cache_reuse	                    | No	      | This feature is only supported for GPT-like model on TRTLLM (as of 0.7.1) and need to compile the model with `--use_paged_context_fmha`. Let the LLM model to remember the last used input KV cache and try to reuse it in the next run. An instant benefit will be blazing fast first token latency. This is typically helpful for document understanding, chat applications that usually have the same input prefix. The TRTLLM backends will remember the prefix tree of the input and reuse most of its part for the next generation. However, this does come with the cost of extra GPU memory. | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.baichuan_model_version	                | No	      | Parameter that exclusively for Baichuan LLM model to specify the version of the model. Need to specify the HF Baichuan checkpoint path. For v1_13b, you should use whether baichuan-inc/Baichuan-13B-Chat or baichuan-inc/Baichuan-13B-Base. For v2_13b, you should use whether baichuan-inc/Baichuan2-13B-Chat or baichuan-inc/Baichuan2-13B-Base. More Baichuan models could be found on baichuan-inc.	                                                                                                                                                                                            | `v1_7b`, `v1_13b`, `v2_7b`, `v2_13b`. <br/> Default is `v1_13b`	                                                                                                            |
| option.chatglm_model_version                  | No             | Parameter exclusive to ChatGLM models to specify the exact model type. Required for ChatGLM models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `chatglm_6b`, `chatglm2_6b`, `chatglm2_6b_32k`, `chatglm3_6b`, `chatglm3_6b_base`, `chatglm3_6b_32k`, `glm_10b`. <br/> Default is `unspecified`, which will throw an error. |
| option.gpt_model_version                  | No             | Parameter exclusive to GPT2 models to specify the exact model type. Required for GPT2 models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `gpt2`, `santacoder`, `starcoder`. <br/> Default is `gpt2`. |
| option.multi_block_mode                  | No             | Split long kv sequence into multiple blocks (applied to generation MHA kernels). It is beneifical when `batch x num_heads` cannot fully utilize GPU. This is **not** supported for qwen model type.                                                                                                                                                                                                                                                                                                                                                                                                  | `true`, `false`. <br/> Default is `false`	 |
| option.use_fused_mlp                  | No             | Enable horizontal fusion in GatedMLP, reduces layer input traffic and potentially improves performance for large Llama models(e.g. llama-2-70b). This option is only supported for Llama model type.                                                                                                                                                                                                                                                                                                                                                                                                 | `true`, `false`. <br/> Default is `false`	 |
| option.rotary_base                  | No             | Rotary base parameter for RoPE embedding. This is supported for llama, internlm, qwen model types                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `float` value. <br/> Default is `10000.0`	 |
| option.rotary_dim                | No             | Rotary dimension parameter for RoPE embedding. This is supported for only gptj model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `int` value. <br/> Default is `64`	 |
| option.rotary_scaling_type </br> option.rotary_scaling_factor                | No             | Rotary scaling parameters. These two options should always be set together to prevent errors. These are supported for llama, qwen and internlm models                                                                                                                                                                                                                                                                                                                                                                                                                                                | The value of `rotary_scaling_type` can be either `linear` and `dynamic`. The value of `rotary_scaling_factor` can be any value larger than 1.0. Default is `None`.|
| Advanced parameters: SmmothQuant	 |
| option.quantize	                              | No	      | Currently only supports `smoothquant` for Llama, Mistral, InternLM and Baichuan models with just in time compilation mode.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `smoothquant`	                                                                                                                                                              |
| option.smoothquant_alpha	                     | No	      | smoothquant alpha parameter	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Default value is `0.8`	                                                                                                                                                     |
| option.smoothquant_per_token	                 | No	      | This is only applied when `option.quantize` is set to `smoothquant`.  This enables choosing at run time a custom smoothquant scaling factor for each token. This is usally little slower and more accurate	                                                                                                                                                                                                                                                                                                                                                                                          | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.smoothquant_per_channel	               | No	      | This is only applied when `option.quantize` is set to `smoothquant`.  This enables choosing at run time a custom smoothquant scaling factor for each channel. This is usally little slower and more accurate	                                                                                                                                                                                                                                                                                                                                                                                        | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| option.multi_query_mode	                      | No	      | This is only needed when `option.quantize` is set to `smoothquant` . This is should be set for models that support multi-query-attention, for e.g llama-70b	                                                                                                                                                                                                                                                                                                                                                                                                                                         | `true`, `false`. <br/> Default is `false`	                                                                                                                                  |
| Advanced parameters: AWQ	                     |
| option.quantize	                              | No	      | Currently only supports `awq` for Llama and Mistral models with just in time compilation mode.	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `awq`	                                                                                                                                                                      |
| option.awq_format	                     | No	      | This is only applied when `option.quantize` is set to `awq`. awq format you want to set. Currently only support `int4_awq`                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Default value is `int4_awq`	                                                                                                                                                |
| option.awq_calib_size	                 | No	      | This is only applied when `option.quantize` is set to `awq`. Number of samples for calibration. 	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Default is `32`	                                                                                                                                                            |


