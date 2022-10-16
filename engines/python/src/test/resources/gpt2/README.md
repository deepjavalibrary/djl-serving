# Deploy Huggingface large model with DeepSpeed

DJLServing has built-in support for DeepSpeed. The following is an example to serve GPT2 model.

## Prepare `serving.properties`

First, we create a file called `serving.properties` that contains information about how to load your model:
- Which DJL Engine to use, in this case, we use `DeepSpeed`
- The number number of GPUs (model slicing number) to shard the model: `tensor_parallel_degree`
- The HuggingFace `model_id`, this is optional, if `model_id` is not defined, DJL will load model
from the model directory.
- The python model `entryPoint` (optional): define which built-in model loading handler to use.
In this example we use `djl_python.deepspeed`. If `entryPoint` is not defined, DJL will look for
custom model handler (`model.py`) in the model directory.
- The maximum output tokens `max_new_tokens` (optional, default 50). This value can be override per
inference request.
- If load the worker in parallel `parallel_loading` (optional, default false). Set to `true` to
reduce model loading time if your model can fit in CPU memory with multiple processes.
- Number of workers `minWorkers` and `maxWorkers` (optional). DJL will auto-detect for you based on
the number of GPUs you have.

```
engine=DeepSpeed
option.entryPoint=djl_python.deepspeed
option.parallel_loading=true
option.tensor_parallel_degree=2
option.model_loading_timeout=600
option.model_id=gpt2
#option.max_new_tokens=50
#minWorkers=1
#maxWorkers=1
```

## Serve your model

```shell
cd djl-serving/engines/python

docker run --runtime=nvidia -it -v src/test/resources/gpt2:/opt/ml/model/gpt2 \ 
-p 8080:8080 deepjavalibrary/djl-serving:deepspeed-nightly

# or
docker run --runtime=nvidia -it -v $PWD:/work \ 
-p 8080:8080 deepjavalibrary/djl-serving:deepspeed-nightly djl-serving -m /work/src/test/resources/gpt2
```

## Run inference

The built-in DeepSpeed handler accepts both `application/json` and `text/plain` content-type
as input. You can also run inference with small batches.

```shell
curl -X POST http://localhost:8080/predictions/gpt2 -H "Content-type: text/plain" -d "Deepspeed is"
```

### predict wth batch

```shell
curl -X POST http://127.0.0.1:8080/predictions/gpt2 -H "Content-type: application/json" -d "{'inputs':['Deepspeed is', 'Example text']}"
# or
curl -X POST http://127.0.0.1:8080/predictions/gpt2 -H "Content-type: application/json" -d "['Deepspeed is', 'Example text']"
```

### predict with different output tokens

```shell
curl -X POST http://127.0.0.1:8080/predictions/gpt2 -H "Content-type: application/json" -d "{'max_new_tokens':250, 'inputs':'Deepspeed is'}"
```

## Customize your model

If you have a non-standard HuggingFace model, you can supply your customized handler code with
a `model.py` in the model directory. You can use [model.py](model.py) as template. You can also
referFor to the [build-in DeepSpeed handler](../../../../setup/djl_python/deepspeed.py) for handing
complicated input data.
