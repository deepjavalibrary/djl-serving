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
- XGBoost model
- LightGBM model
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
curl -O https://publish.djl.ai/djl-serving/djl-serving_0.26.0-1_all.deb
sudo dpkg -i djl-serving_0.26.0-1_all.deb
```

For Windows

We are considering to create a `chocolatey` package for Windows. For the time being, you can
download djl-serving zip file from [here](https://publish.djl.ai/djl-serving/serving-0.26.0.zip).

```
curl -O https://publish.djl.ai/djl-serving/serving-0.26.0.zip
unzip serving-0.26.0.zip
# start djl-serving
serving-0.26.0\bin\serving.bat
```

### Docker

You can also use docker to run DJL Serving:

```
docker run -itd -p 8080:8080 deepjavalibrary/djl-serving
```

## Usage

DJL Serving can be started from the command line.
To see examples, see the [starting page](serving/docs/starting.md).

### More examples

- [Serving a Python model](https://github.com/deepjavalibrary/djl-demo/tree/master/huggingface/python)
- [Serving on Inferentia EC2 instance](https://github.com/deepjavalibrary/djl-demo/tree/master/huggingface/inferentia)
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

Details about how DJL Serving is implemented can be found in the [architecture docs](serving/docs/architecture.md).

# Plugin management

DJL Serving supports plugins, user can implement their own plugins to enrich DJL Serving features.
See [DJL Plugin Management](serving/docs/plugin_management.md) for how to install plugins to DJL Serving.

## Logging
you can set the logging level on the command-line adding a parameter for the JVM

```sh
-Dai.djl.logging.level={FATAL|ERROR|WARN|INFO|DEBUG|TRACE}
```
