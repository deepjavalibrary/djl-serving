# djl-serving docker image

This module contains docker file to build djl-serving docker image. This docker image
is compatible with SageMaker hosting.

## Build docker image

Run
```shell
cd djl-serving/serving/docker

# build djl-serving cpu docker image
docker build -t deepjavalibrary/djl-serving:0.17.0 --build-arg djl_version=0.17.0 .

# build docker image for inferentia
docker build -t deepjavalibrary/djl-serving:0.17.0-inf1 --build-arg version=0.17.0 -f inf1.Dockerfile .
```

## Run docker image

**djl-serving** will load all models stored in `/opt/ml/model` folder. You only need to
download your model files and mount the model folder to `/opt/ml/model` in the docker container.

Here is an example to run djl-serving docker image:

```shell
mkdir models
cd models
curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_jit.tar.gz

docker run -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.17.0
```

Here is an example to run djl-serving with AWS Inferentia:

```shell
mkdir models
cd models

curl -O https://resources.djl.ai/test-models/pytorch/bert_qa_inf1.tar.gz
docker run --device /dev/neuron0 -it --rm -v $PWD:/opt/ml/model -p 8080:8080 deepjavalibrary/djl-serving:0.17.0-inf1
```
