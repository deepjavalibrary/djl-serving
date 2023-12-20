# Model Configuration

You set per model settings by adding a serving.properties file in the root of your model directory (or .zip).
These apply for all engines and modes.

An example `serving.properties` can be found [here](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/src/test/resources/identity/serving.properties).

## Main properties

In `serving.properties`, you can set the following properties. Model properties are accessible to `Translator`
and python handler functions.

- `engine`: Which Engine to use, values include MXNet, PyTorch, TensorFlow, ONNX, PaddlePaddle, DeepSpeed, etc.
- `load_on_devices`: A ; delimited devices list, which the model to be loaded on, default to load on all devices.
- `translatorFactory`: Specify the TranslatorFactory.
- `job_queue_size`: Specify the job queue size at model level, this will override global `job_queue_size`, default is `1000`.
- `batch_size`: the dynamic batch size, default is `1`.
- `max_batch_delay` - the maximum delay for batch aggregation in millis, default value is `100` milliseconds.
- `max_idle_time` - the maximum idle time in seconds before the worker thread is scaled down, default is `60` seconds.
- `log_model_metric`: Enable model metrics (inference, pre-process and post-process latency) logging.
- `metrics_aggregation`: Number of model metrics to aggregate, default is `1000`.
- `minWorkers`: Minimum number of workers, default is `1`.
- `maxWorkers`: Maximum number of workers, default is `#CPU/OMP_NUM_THREAD` for CPU, GPU default is `2`, inferentia default is `2` (PyTorch engine), `1` (Python engine) .
- `gpu.minWorkers`: Minimum number of workers for GPU.
- `gpu.maxWorkers`: Maximum number of workers for GPU.
- `cpu.minWorkers`: Minimum number of workers for CPU.
- `cpu.maxWorkers`: Maximum number of workers for CPU.
- `required_memory_mb`: Specify the required memory (CPU and GPU) in MB to load the model.
- `gpu.required_memory_mb`: Specify the required GPU memory in MB to load the model.
- `reserved_memory_mb`: Reserve memory in MB to avoid system out of memory.
- `gpu.reserved_memory_mb`: Reserve GPU memory in MB to avoid system out of memory.

## Option Properties

In `serving.properties`, you can also set options (prefixed with `option`) and properties.
The options will be passed to `Model.load(Path modelPath, String prefix, Map<String, ?> options)` API.
It allows you to set engine specific configurations.
Here are some of the available option properties:

```
# set model file name prefix if different from folder name
option.modeName=resnet18_v1

# PyTorch options
option.mapLocation=true
option.extraFiles=foo.txt,bar.txt

# ONNXRuntime options
option.interOpNumThreads=2
option.intraOpNumThreads=2
option.executionMode=SEQUENTIAL
option.optLevel=BASIC_OPT
option.memoryPatternOptimization=true
option.cpuArenaAllocator=true
option.disablePerSessionThreads=true
option.customOpLibrary=myops.so
option.disablePerSessionThreads=true
option.ortDevice=TensorRT/ROCM/CoreML

# Python model options
# Mark model as failure after python process crashing 10 times
retry_threshold=10

option.pythonExecutable=python3
option.entryPoint=deepspeed.py
option.handler=hanlde
option.predict_timeout=120
option.model_loading_timeout=10
option.parallel_loading=true
option.tensor_parallel_degree=2
option.enable_venv=true
option.rolling_batch=auto
#option.rolling_batch=lmi-dist
option.max_rolling_batch_size=64
option.paged_attention=false
option.max_rolling_batch_prefill_tokens=1088

# max output size in bytes, default to 60M
option.max_output_size=67108864
```

Most of the options can also be overriden by an environment variable with the `OPTION_` prefix and all caps.
For example:

```
# to enable rolling batch with only environment variable:
export OPTION_ROLLING_BATCH=auto
```

## Basic Model Configurations

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

job queue size, batch size, max batch delay, max worker idle time can be configured at
per model level, this will override global settings:

```
job_queue_size=10
batch_size=2
max_batch_delay=1
max_idle_time=120
```

You can configure which device to load the model on, default is *:

```
load_on_devices=gpu4;gpu5
# or simply:
load_on_devices=4;5
```

## Python model configuration

#### number of workers

For Python engine, we recommend set `minWorkers` and `maxWorkers` to be the same since python
worker scale up and down is expensive.

You may also need to consider `OMP_NUM_THREAD` when setting number workers. `OMP_NUM_THREAD` is default
to `1`, you can unset `OMP_NUM_THREAD` by setting `NO_OMP_NUM_THREADS=true`. If `OMP_NUM_THREAD` is unset,
the `maxWorkers` will be default to 2 (larger `maxWorkers` with non 1 `OMP_NUM_THREAD` can cause thread
contention, and reduce throughput).

Set minimum workers and maximum workers for your model:

```
minWorkers=32
maxWorkers=64
# idle time in seconds before the worker thread is scaled down
max_idle_time=120
```

Or set minimum workers and maximum workers differently for GPU and CPU:

```
gpu.minWorkers=2
gpu.maxWorkers=3
cpu.minWorkers=2
cpu.maxWorkers=4
```

**Note**: Loading model in Python mode is pretty heavy. We recommend to set `minWorker` and `maxWorker` to be the same value to avoid unnecessary load and unload.


#### job queue size
Or override global `job_queue_size`:

```
job_queue_size=10
```

#### dynamic batching
To enable dynamic batching:

```
batch_size=2
max_batch_delay=1
```

#### rolling batch
To enable rolling batch for Python engine:

```
# lmi-dist and vllm requires running mpi mode
engine=MPI
option.rolling_batch=auto
# use FlashAttention
#option.rolling_batch=lmi-dist
#option.rolling_batch=scheduler
option.max_rolling_batch_size=64

# increase max_rolling_batch_prefill_tokens for long sequence
option.max_rolling_batch_prefill_tokens=1088

# disable PagedAttention if run into OOM
option.paged_attention=false
```

## Appendix

### How to download uncompressed model from S3
To enable fast model downloading, you can store your model artifacts (weights) in a S3 bucket, and
only keep the model code and metadata in the `model.tar.gz` (.zip) file. DJL can leverage
[s5cmd](https://github.com/peak/s5cmd) to download uncompressed files from S3 with extremely fast
speed.

To enable `s5cmd` downloading, you can configure `serving.properties` as the following:

```
option.model_id=s3://YOUR_BUCKET/...
```

### How to resolve python package conflict between models
If you want to deploy multiple python models, but their dependencies has conflict, you can enable
[python virtual environments](https://docs.python.org/3/tutorial/venv.html) for your model:

```
option.enable_venv=true
```

