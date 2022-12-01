# DJL Serving configurations

DJL serving is highly configurable. This document tries to capture those configurations in a single document.

## DJL settings

DJLServing build on top of Deep Java Library (DJL). Here is a list of settings for DJL:

| Key                            | Type        | Description                                                                         |
|--------------------------------|-------------|-------------------------------------------------------------------------------------|
| DJL_DEFAULT_ENGINE             | env var     | The preferred engine for DJL if there are multiple engines, default: MXNet          |
| ai.djl.default_engine          | system prop | The preferred engine for DJL if there are multiple engines, default: MXNet          |
| DJL_CACHE_DIR                  | env var     | The cache directory for DJL: default: $HOME/.djl.ai/                                |
| ENGINE_CACHE_DIR               | env var     | The cache directory for engine native libraries: default: $DJL_CACHE_DIR            |
| ai.djl.dataiterator.autoclose  | system prop | Automatically close data set iterator, default: true                                |
| ai.djl.repository.zoo.location | system prop | global model zoo search locations, not recommended                                  |
| offline                        | system prop | Don't access network for downloading engine's native library and model zoo metadata |
| collect-memory                 | system prop | Enable memory metric collection, default: false                                     |
| disableProgressBar             | system prop | Disable progress bar, default: false                                                |

### PyTorch

| Key                                | Type        | Description                                                                                                                                                                          |
|------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PYTORCH_LIBRARY_PATH	              | env var     | User provided custom PyTorch native library                                                                                                                                          |
| PYTORCH_VERSION                    | env var     | PyTorch version to load                                                                                                                                                              |
| PYTORCH_EXTRA_LIBRARY_PATH         | env var	    | Custom pytorch library to load (e.g. torchneuron/torchvision/torchtext)                                                                                                              |
| PYTORCH_PRECXX11                   | env var	    | Load precxx11 libtorch                                                                                                                                                               |
| PYTORCH_FLAVOR                     | env var	    | To force override auto detection (e.g. cpu/cpu-precxx11/cu102/cu116-precxx11)                                                                                                        |
| PYTORCH_JIT_LOG_LEVEL              | env var	    | Enable [JIT logging](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/jit_log.h)                                                                                        |
| ai.djl.pytorch.native_helper	      | system prop | A user provided custom loader class to help locate pytorch native resources                                                                                                          |
| ai.djl.pytorch.num_threads         | system prop | Override OMP_NUM_THREAD environment variable                                                                                                                                         |
| ai.djl.pytorch.num_interop_threads | system prop | Set PyTorch interop threads                                                                                                                                                          |
| ai.djl.pytorch.graph_optimizer     | system prop | Enable/Disable JIT execution optimize, default: true. See: https://github.com/deepjavalibrary/djl/blob/master/docs/development/inference_performance_optimization.md#graph-optimizer |
| ai.djl.pytorch.cudnn_benchmark     | system prop | To speed up ConvNN related model loading, default: false                                                                                                                             |
| ai.djl.pytorch.use_mkldnn          | system prop | Enable MKLDNN, default: false, not recommended, use with your own risk                                                                                                               |

### TensorFlow

| Key                         | Type    | Description                                       |
|-----------------------------|---------|---------------------------------------------------|
| TENSORFLOW_LIBRARY_PATH     | env var | User provided custom TensorFlow native library    |
| TENSORRT_EXTRA_LIBRARY_PATH | env var | Extra TensorFlow custom operators library to load |
| TF_CPP_MIN_LOG_LEVEL        | env var | TensorFlow log level                              |
| ai.djl.tensorflow.debug     | env var | Enable devicePlacement logging, default: false    |

### MXNet

| Key                               | Type        | Description                                                                    |
|-----------------------------------|-------------|--------------------------------------------------------------------------------|
| MXNET_LIBRARY_PATH                | env var     | User provided custom MXNet native library                                      |
| MXNET_VERSION                     | env var     | The version of custom MXNet build                                              |
| MXNET_EXTRA_LIBRARY_PATH          | env var     | Load extra MXNet custom libraries, e.g. Elastice Inference                     |
| MXNET_EXTRA_LIBRARY_VERBOSE       | env var     | Set verbosity for MXNet custom library                                         |
| ai.djl.mxnet.static_alloc         | system prop | CachedOp options, default: true                                                |
| ai.djl.mxnet.static_shape         | system prop | CachedOp options, default: true                                                |
| ai.djl.use_local_parameter_server | system prop | Use java parameter server instead of MXNet native implemention, default: false |

### PaddlePaddle

| Key                                     | Type        | Description                                      |
|-----------------------------------------|-------------|--------------------------------------------------|
| PADDLE_LIBRARY_PATH                     | env var     | User provided custom PaddlePaddle native library |
| ai.djl.paddlepaddle.disable_alternative | system prop | Disable alternative engine                       |

### Neo DLR (TVM)

| Key                            | Type        | Description                             |
|--------------------------------|-------------|-----------------------------------------|
| DLR_LIBRARY_PATH               | env var     | User provided custom DLR native library |
| ai.djl.dlr.disable_alternative | system prop | Disable alternative engine              |

### Python

| Key                                | Type        | Description                                             |
|------------------------------------|-------------|---------------------------------------------------------|
| PYTHON_EXECUTABLE                  | env var	    | The location is python executable, default: python      |
| DJL_ENTRY_POINT                    | env var	    | The entrypoint python file or module, default: model.py |
| MODEL_LOADING_TIMEOUT              | env var	    | Python worker load model timeout: default: 240 seconds  |
| PREDICT_TIMEOUT                    | env var	    | Python predict call timeout, default: 120 seconds       |
| ai.djl.python.disable_alternative  | system prop | Disable alternative engine                              |

### Python (DeepSpeed)

| Key                    | Type     | Description                                          |
|------------------------|----------|------------------------------------------------------|
| TENSOR_PARALLEL_DEGREE | env var	 | Required<br>Set tensor parallel degree for DeepSpeed |

## Global Model Server settings

Global settings are configured at model server level. Change to these settings usually requires
restart model server to take effect.

Most of the model server specific configuration can be configured in `conf/config.properties` file.
You can find the configuration keys here:
[ConfigManager.java](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/src/main/java/ai/djl/serving/util/ConfigManager.java#L52-L79)

Each configuration key can also be override by environment variable with `SERVING_` prefix, for example:

```
export SERVING_JOB_QUEUE_SIZE=1000 # This will override JOB_QUEUE_SIZE in the config
```

| Key               | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MODEL_SERVER_HOME | env var | DJLServing home directory, default: Installation directory (e.g. /usr/local/Cellar/djl-serving/0.19.0/)                                                                                                                                                                                                                                                                                                                                                   |
| DEFAULT_JVM_OPTS  | env var | default: `-Dlog4j.configurationFile=${APP_HOME}/conf/log4j2.xml`<br>Override default JVM startup options and system properties.                                                                                                                                                                                                                                                                                                                           |
| JAVA_OPTS         | env var | default: `-XX:+UseContainerSupport -XX:+ExitOnOutOfMemoryError`<br>Add extra JVM options.                                                                                                                                                                                                                                                                                                                                                                 |
| SERVING_OPTS      | env var | default: N/A<br>Add serving related JVM options.<br>Some of DJL configuration can only be configured by JVM system properties, user has to set DEFAULT_JVM_OPTS environment variable to configure them.<br>- `-Dai.djl.pytorch.num_interop_threads=2`, this will override interop threads for PyTorch<br>- `-Dai.djl.pytorch.num_threads=2`, this will override OMP_NUM_THREADS for PyTorch<br>- `-Dai.djl.logging.level=debug` change DJL loggging level |

## Model specific settings

You set per model settings by adding a `serving.properties` file in the root of your model directory (or .zip).

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

### Python (DeepSpeed)

For Python (DeepSpeed) engine, DJL load multiple workers sequentially by default to avoid run
out of memory. You can reduced model loading time by parallel loading workers if you know the
peak memory won’t cause out of memory:

```
parallel_loading=true
# Allows to load DeepSpeed workers in parallel
option.parallel_loading=true
# specify tensor parallel degree (number of partitions)
option.tensor_parallel_degree=2
# specify per model timeout
option.model_loading_timeout=600
option.predict_timeout=240

# use built-in DeepSpeed handler
option.entryPoint=djl_python.deepspeed
# passing extra options to model.py or built-in handler
option.model_id=gpt2
option.data_type=fp32
option.max_new_tokens=50

# defines custom environment variables
env=LARGE_TENSOR=1
# specify the path to the python executable
pythonExecutable=/usr/bin/python3
```

## Engine specific settings

DJL support 12 deep learning frameworks, each framework has their own settings. Please refer to
each framework’s document for detail.

A common setting for most of the engines is ``OMP_NUM_THREADS``, for the best throughput,
DJLServing set this to 1 by default. For some engines (e.g. **MXNet**, this value must be one).
Since this is a global environment variable, setting this value will impact all other engines.

The follow table show some engine specific environment variables that is override by default by DJLServing:

| Key                    | Engine     | Description                                         |
|------------------------|------------|-----------------------------------------------------|
| TF_NUM_INTEROP_THREADS | TensorFlow | default 1, OMP_NUM_THREADS will override this value |
| TF_NUM_INTRAOP_THREADS | TensorFlow | default 1                                           |
| TF_CPP_MIN_LOG_LEVEL	  | TensorFlow | default 1                                           |
| TVM_NUM_THREADS        | DLR/TVM    | default 1, OMP_NUM_THREADS will override this value |
| MXNET_ENGINE_TYPE      | MXNet      | this value must be `NaiveEngine`                    |

## Appendix

### How to configure logging

#### Option 1: enable debug log:

```
export SERVING_OPTS="-Dai.djl.logging.level=debug"
```

#### Option 2: use your log4j2.xml

```
export DEFAULT_JVM_OPTS="-Dlog4j.configurationFile=/MY_CONF/log4j2.xml
```
