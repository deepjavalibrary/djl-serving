# DJL Serving Model Packaging

## Overview

DJL Serving is a high-performance serving system for deep learning models. DJL Serving supports models with Python mode, Java mode, as well as binary mode.

## Properties file

The serving.properties is a configuration file that can be used in all modes.

You can set number of workers for each model:
https://github.com/deepjavalibrary/djl-serving/blob/master/serving/src/test/resources/identity/serving.properties#L4-L8

For example, set minimum workers and maximum workers for your model:

```
minWorkers=32
maxWorkers=64
```

Or you can configure minimum workers and maximum workers differently for GPU and CPU:

```
gpu.minWorkers=2
gpu.maxWorkers=3
cpu.minWorkers=2
cpu.maxWorkers=4
```

job queue size can be configured at per model level, this will override global `job_queue_size`:

```
job_queue_size=10
```

## Python

This section walks through how to serve Python based model with DJL Serving.

### Define a Model

To get started, implement a python source file named `model.py` as the entry point. DJL Serving will run your request by invoking a `handle` function that you provide. The `handle` function should have the following signature:

```
def handle(inputs: Input)
```

If there are other packages you want to use with your script, you can include a `requirements.txt` file in the same directory with your model file to install other dependencies at runtime.

A `requirements.txt` file is a text file that contains a list of items that are installed by using pip install. You can also specify the version of an item to install.

If you don't want to install package from internet, you can bundle the python installation wheel in the model directory and install the package from model directory:

```
./local_wheels/ABC-0.0.2-py3-none-any.whl
```

### Packaging

DJL Serving supports model artifacts in model directory, .zip or .tar.gz format.

To package model artifacts in a .zip:

```
zip model.zip /path/to/model
```

To package model artifacts in a .tar.gz:

```
tar -czvf model.tar.gz /path/to/model
```

### Serving Example

Let's run an example where we load a model and run inference using the REST API.

#### Step 1: Start Server

First, start the server and load a model at startup. We will use the resnet18 model as an example. To get this model, first clone the djl-demo repo.

```
git clone https://github.com/deepjavalibrary/djl-demo.git
cd djl-demo
```

The resnet18 model is located at djl-serving/python-mode/resnet18. It provides a [model.py](https://github.com/deepjavalibrary/djl-demo/blob/master/djl-serving/python-mode/resnet18/model.py) that implements a `handle` function.

```
def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
```

It also provides a `requirements.txt` that loads torchvision 0.12.0:

```
torchvision==0.12.0
```

Next, run the DJL Serving and load this model at startup.

Linux/macOS

```
djl-serving -m resnet::Python=file://$PWD/djl-serving/python-mode/resnet18
```

Windows

```
path-to-your\serving.bat -m "resnet::Python=file:///%cd%\djl-serving\python-mode\resnet18"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `resnet` with the model.

#### Step 2: Inference

To query the model using the prediction API, open another session and run the following command:

Linux/macOS

```
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST "http://127.0.0.1:8080/predictions/resnet" -T "kitten.jpg"
```

On Windows, you can just download the image and use `Postman` to send POST request.

This should return the following result:

```json
[
  {
    "tabby":0.4552347958087921,
    "tiger_cat":0.3483535945415497,
    "Egyptian_cat":0.15608155727386475,
    "lynx":0.026761988177895546,
    "Persian_cat":0.002232028404250741
  }
]
```
