#  LMI environment variable instruction

LMI allow customer to specify environment variable. For example, if I want to deploy a LLM model with LMI without creating any files. Here is some options you can do

## Standalone serving.properties to Environment variable

option.model_id in LMI can be a HuggingFace Model ID, or an S3 url point to an uncompressed folder

serving.properties

```
engine=MPI
option.model_id=tiiuae/falcon-40b
option.entryPoint=djl_python.transformersneuronx
option.trust_remote_code=true
option.tensor_parallel_degree=4
option.max_rolling_batch_size=32
option.rolling_batch=auto
```

The above serving.properties can be translated into all environment variable settings

```
HF_MODEL_ID=tiiuae/falcon-40b
HF_TRUST_REMOTE_CODE=true
TENSOR_PARALLEL_DEGREE=4
OPTION_ENTRYPOINT=djl_python.transformersneuronx
OPTION_MAX_ROLLING_BATCH_SIZE=32
OPTION_ROLLING_BATCH=auto
```

engine translate from

```
engine=<engine name> -> SERVING_LOAD_MODELS=test::<engine name>=/opt/ml/model
```

All the rest properties are translate as

```
option.<properties> -> OPTION_<PROPERTIES>
```

If there are properties not starting with option, those are typically model server parameter, you can specify as following

```
batch_size=4
max_batch_delay=200
```

Those are translate into

```
SERVING_BATCH_SIZE=4
SERVING_MAX_BATCH_DELAY=200
```

## SageMaker trained model translation

Let’s assume we used SageMaker to train/fine-tuned a model and upload to S3 as a tar.gz file.
The file is located in:

```
s3://my-training-repo/my_fine_tuned_llama.tar.gz
```

Assume this file is a standard HuggingFace saved model. Here are something you can set to not alter the file to its original format

```
from sagemaker import Model
code_artifact = s3://my-training-repo/my_fine_tuned_llama.tar.gz
env = {"HF_TASK": "text-generation",
       "TENSOR_PARALLEL_DEGREE": 4,
       "OPTION_ROLLING_BATCH": "auto"}
model = Model(image_uri=image_uri, model_data=code_artifact, role=role)
```

In this case, we will build the serving.properties on the fly for you and no other coding required!
