# djl-serving docker image

This module contains docker file to build djl-serving docker image. This docker image
is compatible with SageMaker hosting.

## Build docker image

Currently, we created docker compose to simplify the building experience. Just run

```shell
cd serving/docker
export DJL_VERSION=$(cat ../../gradle.properties | awk -F '=' '/djl_version/ {print $2}')
docker compose build --build-arg djl_version=${DJL_VERSION} <compose-target>
```

You can find different `compose-target` in `docker-compose.yml`, like `cpu`, `deepspeed`...

## Run docker image

DJLServing is doing nightly publish to the [dockerhub](https://hub.docker.com/r/deepjavalibrary/djl-serving/tags).
You can just pull the image you need from there.

**djl-serving** will load all models stored in `/opt/ml/model` folder. You only need to
download your model files and mount the model folder to `/opt/ml/model` in the docker container.

Here are a few examples to run djl-serving docker image:

### CPU

```shell
mkdir models
cd models
curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_jit.tar.gz

docker run -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.25.0
```

### GPU

```shell
mkdir models
cd models
curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_jit.tar.gz

docker run -it --runtime=nvidia --shm-size 2g -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.25.0-pytorch-gpu
```

### AWS Inferentia

```shell
mkdir models
cd models

curl -O https://resources.djl.ai/test-models/pytorch/resnet18_inf2_2_4.tar.gz
docker run --device /dev/neuron0 -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.25.0-pytorch-inf2
```
