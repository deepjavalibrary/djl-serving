# DJL Serving

## Overview

DJL Serving is a high performance universal stand-alone model serving solution powered by [DJL](https://djl.ai).
It takes a deep learning model, several models, or workflows and makes them available through an
HTTP endpoint. It can serve the following model types out of the box:

- PyTorch TorchScript model
- TensorFlow SavedModel bundle
- Apache MXNet model
- ONNX model (CPU)
- TensorRT model
- Python script model

You can install extra extensions to enable the following models:

- PaddlePaddle model
- TFLite model
- Neo DLR (TVM) model
- XGBoost model
- Sentencepiece model
- fastText/BlazingText model

## Key features

- **Performance** - DJL serving running multithreading inference in a single JVM. Our benchmark shows
DJL serving has higher throughput than most C++ model servers on the market
- **Ease of use** - DJL serving can serve most models out of the box
- **Easy to extend** - DJL serving plugins make it easy to add custom extensions
- **Auto-scale** - DJL serving automatically scales up/down worker threads based on the load
- **Dynamic batching** - DJL serving supports dynamic batching to increase throughput
- **Model versioning** - DJL allows users to load different versions of a model on a single endpoint
- **Multi-engine support** - DJL allows users to serve models from different engines at the same time

## Installation

For macOS

```
brew install djl-serving

# Start djl-serving as service:
brew services start djl-serving

# Stop djl-serving service
brew services stop djl-serving
```

For Ubuntu

```
curl -O https://publish.djl.ai/djl-serving/djl-serving_0.18.0-1_all.deb
sudo dpkg -i djl-serving_0.18.0-1_all.deb
```

For Windows

We are considering to create a `chocolatey` package for Windows. For the time being, you can 
download djl-serving zip file from [here](https://publish.djl.ai/djl-serving/serving-0.18.0.zip).

```
curl -O https://publish.djl.ai/djl-serving/serving-0.18.0.zip
unzip serving-0.18.0.zip
# start djl-serving
serving-0.18.0\bin\serving.bat
```

### Docker

You can also use docker to run DJL Serving:

```
docker run -itd -p 8080:8080 deepjavalibrary/djl-serving
```

## Usage

### Sample Usage

Use the following command to start model server locally:

```sh
djl-serving
```

The model server will be listening on port 8080. You can also load a model for serving on start up:

```sh
djl-serving -m "https://resources.djl.ai/demo/mxnet/resnet18_v1.zip"
```

Open another terminal, and type the following command to test the inference REST API:

```sh
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST http://localhost:8080/predictions/resnet18_v1 -T kitten.jpg

or:

curl -X POST http://localhost:8080/predictions/resnet18_v1 -F "data=@kitten.jpg"

[
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.4838452935218811
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.20599420368671417
  },
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.18810515105724335
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.06411745399236679
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.010215568356215954
  }
]
```

### Examples for loading models

```shell
# Load models from the DJL model zoo on startup
djl-serving -m "djl://ai.djl.pytorch/resnet"

# Load version v1 of a PyTorch model on GPU(0) from the local file system
djl-serving -m "resnet:v1:PyTorch:0=file:$HOME/models/pytorch/resnet18/"

# Load a TensorFlow model from TFHub
djl-serving -m "resnet=https://tfhub.dev/tensorflow/resnet_50/classification/1"
```

### Examples for customizing data processing

```shell
# Use the default data processing for a well-known application
djl-serving -m "file:/resnet?application=CV/image_classification"

# Specify a custom data processing with a Translator
djl-serving -m "file:/resnet?translatorFactory=MyFactory"

## Pass parameters for data processing
djl-serving -m "djl://ai.djl.pytorch/resnet?applySoftmax=false"
```

### Using DJL Extensions

```shell
# Load a model from an AWS S3 Bucket
djl-serving -m "s3://djl-ai/demo/resnet/resnet18.zip"

# Load a model from HDFS
djl-serving -m "hdfs://localhost:50070/models/pytorch/resnet18/"

# Use a HuggingFace tokenizer
djl-serving -m "file:/resnet?transaltorFactory=ai.djl.huggingface.BertQATranslator"
```

### More examples

- [Serving a Python model](https://github.com/deepjavalibrary/djl-demo/tree/master/huggingface/python)
- [Serving on Inf1 EC2 instance](https://github.com/deepjavalibrary/djl-demo/tree/master/huggingface/inferentia)
- [Serving with docker](https://github.com/deepjavalibrary/djl-serving/tree/master/serving/docker)

### More command line options

```sh
djl-serving --help
usage: djl-serving [OPTIONS]
 -f,--config-file <CONFIG-FILE>    Path to the configuration properties file.
 -h,--help                         Print this help.
 -m,--models <MODELS>              Models to be loaded at startup.
 -s,--model-store <MODELS-STORE>   Model store location where models can be loaded.
 -w,--workflows <WORKFLOWS>   Workflows to be loaded at startup.
```

See [configuration](serving/docs/configuration.md) for more details about defining models, model-store, and workflows.

## REST API

DJL Serving uses a RESTful API for both inference and management calls.

When DJL Serving starts up, it has two web services:

* [Inference API](serving/docs/inference_api.md) - Used by clients to query the server and run models
* [Management API](serving/docs/management_api.md) - Used to add, remove, and scale models on the server

By default, DJL Serving listens on port 8080 and is only accessible from localhost.
Please see [DJL Serving Configuration](serving/docs/configuration.md) for how to enable access from a remote host.

## Architecture

DJL serving is built on top of [Deep Java Library](https://djl.ai). You can visit the
[DJL github repository](https://github.com/deepjavalibrary/djl) to learn more.

It is also possible to leverage only the worker thread pool using the separate [WorkLoadManager](wlm) module.
The separate WorkLoadManager can be used to take advantage of DJL serving's model batching
and threading and integrate it into a custom Java service.

![Architecture Diagram](https://resources.djl.ai/images/djl-serving/architecture.png)

# Plugin management

DJL Serving supports plugins, user can implement their own plugins to enrich DJL Serving features.
See [DJL Plugin Management](serving/docs/plugin_management.md) for how to install plugins to DJL Serving.

## Logging
you can set the logging level on the command-line adding a parameter for the JVM

```sh
-Dai.djl.logging.level={FATAL|ERROR|WARN|INFO|DEBUG|TRACE}
```
