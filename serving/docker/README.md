# djl-serving docker image

This module contains docker file to build djl-serving docker image. This docker image
is compatible with SageMaker hosting.

## Build docker image

Currently, we created docker compose to simplify the building experience. Just run

```shell
cd serving/docker
export DJL_VERSION=$(awk -F '=' '/djl / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)
docker compose build --build-arg djl_version=${DJL_VERSION} <compose-target>
```

You can find different `compose-target` in `docker-compose.yml`, like `cpu`, `lmi`...

## Run docker image

You can find DJL latest release docker image on [dockerhub](https://hub.docker.com/r/deepjavalibrary/djl-serving/tags?page=1&name=0.27.0).
DJLServing also publishes nightly publish to the [dockerhub nightly](https://hub.docker.com/r/deepjavalibrary/djl-serving/tags?page=1&name=nightly).
You can just pull the image you need from there.

**djl-serving** will load all models stored in `/opt/ml/model` folder. You only need to
download your model files and mount the model folder to `/opt/ml/model` in the docker container.

Here are a few examples to run djl-serving docker image:

### CPU

```shell
docker pull deepjavalibrary/djl-serving:0.27.0

mkdir models
cd models
curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_jit.tar.gz

docker run -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.27.0
```

### GPU

```shell
docker pull deepjavalibrary/djl-serving:0.27.0-pytorch-gpu

mkdir models
cd models
curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_jit.tar.gz

docker run -it --runtime=nvidia --shm-size 2g -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.27.0-pytorch-gpu
```

### AWS Inferentia

```shell
docker pull deepjavalibrary/djl-serving:0.27.0-pytorch-inf2

mkdir models
cd models

curl -O https://resources.djl.ai/test-models/pytorch/resnet18_inf2_2_4.tar.gz
docker run --device /dev/neuron0 -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.27.0-pytorch-inf2
```

### aarch64 machine

```shell
docker pull deepjavalibrary/djl-serving:0.27.0-aarch64

mkdir models
cd models

curl -O https://resources.djl.ai/test-models/pytorch/resnet18_inf2_2_4.tar.gz
docker run --device /dev/neuron0 -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.27.0-aarch64
```

## Run docker image with custom command line arguments

You can pass command line arguments to `djl-serving` directly when you using `docker run`

```
docker run -it --rm -p 8080:8080 deepjavalibrary/djl-serving:0.27.0 djl-serving -m "djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2"
```
