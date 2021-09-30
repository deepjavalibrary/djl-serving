# DJL - Python engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for Python based model.

DJL Python engine allows you run python model in a JVM based application. However, you still
need to install your python environment and dependencies.

Python engine is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine](https://docs.djl.ai/docs/hybrid_engine.md).

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.python/python/latest/index.html).

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

- ai.djl.python:python:0.12.0

```xml
<dependency>
    <groupId>ai.djl.python</groupId>
    <artifactId>python</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```
