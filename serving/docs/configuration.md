# DJL Serving Configuration

DJL Serving is a multi-layer system and has many different forms of configuration across those layers.

## Global

At the beginning, there are [global configurations](configurations_global.md).
These configurations are passed through startup arguments, the config file, and environment variables.

As part of the startup, you are able to specify several different categories of options:

- Global Java settings with environment variables like `$JAVA_HOME` and `$JAVA_OPTS`.
- Loading behavior with the `model_store` and what models to load on startup
- Network settings such as the port and SSL

## Engine

DJL Serving is powered by [DeepJavaLibrary](djl.ai) and most of the functionality exists through the use of [DJL engines](http://docs.djl.ai/docs/engine.html).
As part of this, many of the engines along with DJL itself can be configured through the use of environment variables and system properties.

The [engine configuration](configurations.md) document lists these configurations.
These include both the ones global to DJL as well as lists for each engine.
There are configurations for paths, versions, performance, settings, and debugging.
All engine configurations are shared between all models and workers using that engine.

## Workflow

Next, you are able to add and configure a [Workflow](workflows.md).
DJL Serving has a custom solution for handling workflows that is configured through a `workflow.json` or `workflow.yml` file.

## Model

Next, it is possible to specify [model configuration](configurations_model.md).
This is mostly done by using a `serving.properties` file, although there are environment variables that can be used as well.

These configurations are also optional.
If no `serving.properties` is provided, some basic properties such as which engine to use will be inferred.
The rest will back back to the global defaults.

## Application

Alongside the configurations that determine how DJL Serving runs the model, there are also options that can be passed into the model itself.
The primary way is through the [DJL Model](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/Model.html) properties or [DJL Criteria](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/Criteria.html) arguments.
These settings are ultimately dependent on the individual model.
But, here are some documented applications that have additional configurations:

- [Large Language Model Configurations](lmi/configurations_large_model_inference_containers.md)
