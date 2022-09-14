# Benchmark your DL model

dll-bench is a command line tool that makes it easy for you to benchmark the model on different
platforms.

With djl-bench, you can easily compare your model's behavior in different use cases, such as:

- single-threaded vs. multi-threaded
- single input vs. batched inputs
- CPU vs. GPU or other hardware accelerator
- Single GPU vs. multiple GPUs
- default engine options vs. customized engine configuration
- running with different engines
- running with different version of the engine

djl-bench currently support benchmark the following type of models:

- PyTorch TorchScript model
- TensorFlow SavedModel bundle
- Apache MXNet model
- ONNX model
- PaddlePaddle model
- TFLite model
- TensorRT model
- XGBoost model
- Python script model
- Neo DLR (TVM) model

You can build djl-bench from source if you need to benchmark fastText/BlazingText/Sentencepiece models.

## Installation

For Ubuntu

- Install using snap

```
sudo snap install djlbench --classic
sudo snap alias djlbench djl-bench
```

- Or download .deb package from S3

```
curl -O https://publish.djl.ai/djl-bench/0.19.0/djl-bench_0.19.0-1_all.deb
sudo dpkg -i djl-bench_0.19.0-1_all.deb
```

For macOS, centOS or Amazon Linux 2

You can download djl-bench zip file from [here](https://publish.djl.ai/djl-bench/0.19.0/benchmark-0.19.0.zip).

```
curl -O https://publish.djl.ai/djl-bench/0.19.0/benchmark-0.19.0.zip
unzip benchmark-0.19.0.zip
rm benchmark-0.19.0.zip
sudo ln -s $PWD/benchmark-0.19.0/bin/benchmark /usr/bin/djl-bench
```

For Windows

We are considering to create a `chocolatey` package for Windows. For the time being, you can
download djl-bench zip file from [here](https://publish.djl.ai/djl-bench/0.19.0/benchmark-0.19.0.zip).

Or you can run benchmark using gradle:

```
cd djl-serving

gradlew benchmark --args="--help"
```

## Prerequisite

Please ensure Java 8+ is installed and you are using an OS that DJL supported with.

After that, you need to clone the djl project and `cd` into the folder.

DJL supported OS:

- Ubuntu 18.04 and above
- Amazon Linux 2 and above
- MacOS latest version
- Windows 10 (Windows Server 2016+)

If you are trying to use GPU, please ensure the CUDA driver is installed. You can verify that through:

```
nvcc -V
```

to checkout the version. For different Deep Learning engine you are trying to run the benchmark,
they have different CUDA version to support. Please check the individual Engine documentation to ensure your CUDA version is supported.

## Sample benchmark script

Here is a few sample benchmark script for you to refer. You can also skip this and directly follow
the 4-step instructions for your own model.

Benchmark on a Tensorflow model from [tfhub](https://tfhub.dev/) url with all-zeros NDArray input for 10 times:

```
djl-bench -e TensorFlow -u https://tfhub.dev/tensorflow/resnet_50/classification/1 -c 10 -s 1,224,224,3
```

Similarly, this is for PyTorch

```
djl-bench -e PyTorch -u https://alpha-djl-demos.s3.amazonaws.com/model/djl-blockrunner/pytorch_resnet18.zip -n traced_resnet18 -c 10 -s 1,3,224,224
```

Benchmark a model from [ONNX Model Zoo](https://github.com/onnx/models)

```
djl-bench -e OnnxRuntime -u https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz -s 1,3,224,224 -n resnet18v1/resnet18v1 -c 10
```

### Benchmark from ModelZoo

#### MXNet

Resnet50 image classification model:

```
djl-bench -c 2 -s 1,3,224,224 -u djl://ai.djl.mxnet/resnet/0.0.1/resnet50_v2
```

#### PyTorch

SSD object detection model:

```
djl-bench -e PyTorch -c 2 -s 1,3,300,300 -u djl://ai.djl.pytorch/ssd/0.0.1/ssd_300_resnet50
```

## Configuration of Benchmark script

To start your benchmarking, we need to make sure we provide the following information.

- The Deep Learning Engine
- The source of the model
- How many runs you would like to make
- Sample input for the model
- (Optional) Multi-thread benchmark

The benchmark script located [here](https://github.com/deepjavalibrary/djl-serving/blob/master/benchmark/src/main/java/ai/djl/benchmark/Benchmark.java).

Just do the following:

```
djl-bench --help
```

This will print out the possible arguments to pass in:

```
usage: djl-bench [-p MODEL-PATH] -s INPUT-SHAPES [OPTIONS]
 -c,--iteration <ITERATION>               Number of total iterations.
 -d,--duration <DURATION>                 Duration of the test in minutes.
 -e,--engine <ENGINE-NAME>                Choose an Engine for the benchmark.
 -g,--gpus <NUMBER_GPUS>                  Number of GPUS to run multithreading inference.
 -h,--help                                Print this help.
 -l,--delay <DELAY>                       Delay of incremental threads.
    --model-arguments <MODEL-ARGUMENTS>   Specify model loading arguments.
    --model-options <MODEL-OPTIONS>       Specify model loading options.
 -n,--model-name <MODEL-NAME>             Specify model file name.
    --neuron-cores <NEURON-CORES>         Number of neuron cores to run multithreading inference, See
                                          https://awsdocs-neuron.readthedocs-hosted.com.
 -o,--output-dir <OUTPUT-DIR>             Directory for output logs.
 -p,--model-path <MODEL-PATH>             Model directory file path.
 -s,--input-shapes <INPUT-SHAPES>         Input data shapes for the model.
 -t,--threads <NUMBER_THREADS>            Number of inference threads.
 -u,--model-url <MODEL-URL>               Model archive file URL.
```

### Step 1: Pick your deep engine

By default, the above script will use MXNet as the default Engine, but you can always change that by adding the followings:

```
-e TensorFlow # TensorFlow
-e PyTorch # PyTorch
-e MXNet # Apache MXNet
-e PaddlePaddle # PaddlePaddle
-e OnnxRuntime # pytorch
-e TFLite # TFLite
-e TensorRT # TensorRT
-e DLR # Neo DLR
-e XGBoost # XGBoost
-e Python # Python script
```

### Step 2: Identify the source of your model

DJL accept variety of models came from different places.

#### Remote location

Use `--model-url` option to load a model from a URL. The URL must point to an archive file.

The following is a pytorch model

```
-u https://alpha-djl-demos.s3.amazonaws.com/model/djl-blockrunner/pytorch_resnet18.zip
```
We would recommend to make model files in a zip for better file tracking.

#### Local directory

Use `--model-path` option to load model from a local directory or an archive file.

Mac/Linux

```
-p /home/ubuntu/models/pytorch_resnet18
or
-p /home/ubuntu/models/pytorch_resnet18.zip
```

Windows

```
-p C:\models\pytorch_resnet18
or
-p C:\models\pytorch_resnet18.zip
```

If the model file name is different from the parent folder name (or the archive file name), you need
to specify `--model-name` in the `--args`:

```
-n traced_resnet18
```

### Step 3: Define how many runs you would like to make

add `-c` inside with a number

```
-c 1000
```

This will run 1000 times inference.

### Step 4: Define your model inputs

The benchmark script uses dummy NDArray inputs.
It will make fake NDArrays (like `NDArray.ones`) to feed in the model for inference.

If we would like to fake an image:

```
-s 1,3,224,224
```

This will create a NDArray (DataType FLOAT32) of shape(1, 3, 224, 224).

If your model requires multiple inputs like three NDArrays with shape 1, 384 and 384. You can do the followings:

```
-s (1),(384),(384)
```

If you input `DataType` is not FLOAT32, you can specify the data type with suffix:

- f: FLOAT32, this is default and is optional
- s: FLOAT16 (short float)
- d: FLOAT64 (double)
- u: UINT8 (unsigned byte)
- b: INT8 (byte)
- i: INT32 (int)
- l: INT64 (long)
- B: BOOLEAN (boolean)

For example:

```
-s (1)i,(384)f,(384)
```

### Optional Step: multithreading inference

You can also do multi-threading inference with DJL. For example, if you would like to run the inference with 10 threads:

```
-t 10
```

Best thread number for your system: The same number of cores your system have or double of the total cores.

You can also add `-l` to simulate the increment load for your inference server. It will add threads with the delay of time.

```
-t 10 -l 100
```

The above code will create 10 threads with the wait time of 100ms.

## Advanced use cases

For different purposes, we designed different mode you can play with. Such as the following arg:

```
-d 86400
```

This will ask the benchmark script repeatedly running the designed task for 86400 seconds (24 hour).
If you would like to make sure DJL is stable in the long run, you can do that.

You can also keep monitoring the DJL memory usages by enable the following flag:

```
export BENCHMARK_OPTS="-Dcollect-memory=true"
```

The memory report will be made available in `build/memory.log`.
