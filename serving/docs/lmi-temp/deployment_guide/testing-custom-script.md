# Testing for custom script with LMI

Currently, we offer a mechanism to run the LMI python handler in a standalone fashion. It support the following two use cases:

- Testing with serving.properties/environment variables
- Testing with custom model.py

In this tutorial, we will try to create an entrypoint python file `hello_world.py` and demostrate how that could work with our testing APIs.

## Install DJLServing Python module

You can simply do:

### From master branch

```
pip install git+https://github.com/deepjavalibrary/djl-serving.git#subdirectory=engines/python/setup
```

### From a specific DLC version

```
pip install git+https://github.com/deepjavalibrary/djl-serving.git@0.27.0-dlc#subdirectory=engines/python/setup
```

## Running with default handler with rolling batch
You can start a test with DJLServing offered default handlers. Let's create a file called `test.py`:

```python
import os
from djl_python import huggingface
from djl_python.test_model import TestHandler

envs = {
            "HF_MODEL_ID": "NousResearch/Nous-Hermes-Llama2-13b",
            "OPTION_MPI_MODE": "true",
            "OPTION_ROLLING_BATCH": "auto",
            "TENSOR_PARALLEL_DEGREE": "max"
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
```

The model we are trying to run is a LLAMA-13B variant.
Assuming you are on a `g5.12xlarge` machine, you can run the following command to play with handler:

```
mpirun -N 4 --allow-run-as-root python3 test.py
```

## Running with custom model script

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

Execute the above code will run the custom handler
