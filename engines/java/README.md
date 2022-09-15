# DJL - Java engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for Java based model.

DJL Java engine allows you run your customized "Java model". You are required to bundle all packages you use as a fat jar.

Java engine is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. 

To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine](https://docs.djl.ai/docs/hybrid_engine.html).

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.java/java/latest/index.html).

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

- ai.djl.java:java:0.19.0

```xml
<dependency>
    <groupId>ai.djl.java</groupId>
    <artifactId>java</artifactId>
    <version>0.19.0</version>
    <scope>runtime</scope>
</dependency>
```

## Test your java model

TBD
