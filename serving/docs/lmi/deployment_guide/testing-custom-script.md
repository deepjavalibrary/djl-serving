# Testing for custom script/entryPoint with LMI

If you are writing your own custom entryPoint and want to test it before you test it with model server in SageMaker, this tutorial could help with that. 
Currently, we offer a mechanism to run the LMI python handler in a standalone fashion. It supports the following two use cases:
- Testing with serving.properties/environment variables
- Testing with default handlers and see how it produces the output for your input
- Testing with custom model.py

In this guide, we provide two supplementary tutorials:
1. we will try to test our default huggingface handler with rolling batch enabled to auto. 
2. we will try to create an entrypoint python file `hello_world.py` and demonstrate how that could work with our testing APIs.

## Prerequisites

### Step 1: Download and bash into the DLC container

We generally recommend you to do your test based into your DLC container. This is because, lmi-dist needs GLIBC and torch, and you might need to test with the same version as used in the DLC, to avoid facing any problems. You might also need our wheels such as lmi_dist, vllm and tensorrt-llm that we ship into our DLC. 

For example: 

```
docker run -it -p 8080:8080 --shm-size=12g --runtime=nvidia -v /home/ubuntu/test.py:/workplace/test.py \
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.27.0-deepspeed0.12.6-cu121 /bin/bash
```

### Step 2: Install DJLServing Python module

You can simply do:

### From master branch

```
pip install git+https://github.com/deepjavalibrary/djl-serving.git#subdirectory=engines/python/setup
```

### From a specific DLC version

```
pip install git+https://github.com/deepjavalibrary/djl-serving.git@0.27.0-dlc#subdirectory=engines/python/setup
```

## Tutorial 1: Running with default handler with rolling batch
You can start a test with DJLServing offered default handlers. Here we are testing our djl_python.huggingface handler with rolling batch as `lmi-dist`. Let's create a file called `test.py`:

```python
import os
from djl_python import huggingface
from djl_python.test_model import TestHandler

envs = {
            "OPTION_MODEL_ID": "NousResearch/Nous-Hermes-Llama2-13b",
            "OPTION_MPI_MODE": "true",
            "OPTION_ROLLING_BATCH": "lmi-dist",
            "OPTION_TENSOR_PARALLEL_DEGREE": 4
        }

for key, value in envs.items():
    os.environ[key] = value

handler = TestHandler(huggingface)

inputs = [{
    "inputs": "The winner of oscar this year is",
    "parameters": {
        "max_new_tokens": 50
    }
}, {
    "inputs": "A little redhood is",
    "parameters": {
        "max_new_tokens": 256
    }
}]

result = handler.inference_rolling_batch(inputs)
print(result)
```

The model we are trying to run is a LLAMA-13B variant.
Assuming you are on a `g5.12xlarge` machine, you can run the following command to play with handler:

```
mpirun -N 4 --allow-run-as-root python3 test.py
```

Note: Use the `mpirun` only when OPTION_MPI_MODE=true. Otherwise, run it as simple python script. 

## Tutorial 2: Running with custom model script

Let's create a file called `hello_world.py` and this will be the handler for us to use.

```python
from djl_python.outputs import Output

def init_model(properties):
  print(f"Model initialized {properties}")
  return properties

model = None

def handle(inputs):
  global model
  if model == None:
      model = init_model(inputs.get_properties())
  return Output().add(inputs.get_as_json())
```

and create a custom `serving.properties`

```
engine=Python
option.dtype=fp16
option.rolling_batch=auto
option.my_var=value
```


Then we could just do:

```python
import os
import hello_world
from djl_python.test_model import TestHandler, create_json_request

handler = TestHandler(hello_world, os.getcwd())
handler.inference(create_json_request({"Hello": "world"}))
```

Execute the above code will run the custom handler. You can see the properties in your custom handler, that has the custom property `option.var` in your serving.properties.
