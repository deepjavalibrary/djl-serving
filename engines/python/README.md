# DJL - Python engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for Python based model.

DJL Python engine allows you run python model in a JVM based application. However, you still
need to install your python environment and dependencies.

Python engine is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine](https://docs.djl.ai/docs/hybrid_engine.html).

## Documentation

The latest javadocs can be found on [javadoc.io](https://javadoc.io/doc/ai.djl.python/python/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

## Installation
You can pull the Python engine from the central Maven repository by including the following dependency:

- ai.djl.python:python:0.22.0

```xml
<dependency>
    <groupId>ai.djl.python</groupId>
    <artifactId>python</artifactId>
    <version>0.22.0</version>
    <scope>runtime</scope>
</dependency>
```

## Test your python model

Testing python code within Java environment is challenging. We provide a tool to help you develop
and test your python model locally. You can easily use IDE to debug your model.

1. Install djl_python toolkit:

```
cd engines/python/setup
pip install -U -e .
```

2. You can use command line tool or python to run djl model testing. The following command is
an example:

```shell
curl -O https://resources.djl.ai/images/kitten.jpg

djl-test-model --model-dir src/test/resources/resnet18 --input kitten.jpg

# or use python
python -m djl_python.test_model --model-dir src/test/resources/resnet18 --input kitten.jpg
```
