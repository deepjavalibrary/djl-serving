#!/bin/bash

platform=$1 # expected values are "cpu" "cpu-full" "pytorch-cu113" "pytorch-inf1" "aarch64"

rm -rf models
mkdir models && cd models

# all platform models except aarch
general_platform_models_urls=(
"https://resources.djl.ai/test-models/pytorch/resnet18_serving_all.zip"
"https://resources.djl.ai/test-models/tensorflow/resnet50v1.zip"
"https://resources.djl.ai/test-models/onnxruntime/resnet18-v1-7.zip"
"https://resources.djl.ai/test-models/mxnet/ssd_resnet50.zip"
)

# only pytorch models
aarch_models_urls=(
"https://resources.djl.ai/test-models/pytorch/resnet18_serving_all.zip"
)

inf_models_urls=(
"https://resources.djl.ai/test-models/pytorch/resnet18_inf1_1_11.tar.gz"
)

download()
{
  urls=("$@")
  for url in "${urls[@]}"
  do
    filename=${url##*/}
    # does not download the file, if file already exists
    if ! [ -f "${filename}" ]; then
      curl -O "$url"
    fi
  done
}

case $platform in
cpu|cpu-full|pytorch-cu113)
  download "${general_platform_models_urls[@]}"
  ;;
pytorch-inf1)
  download "${inf_models_urls[@]}"
  ;;
aarch64)
  download "${aarch_models_urls[@]}"
  ;;
*)
  echo "Bad argument. Expecting one of the values: cpu, cpu-full, pytorch-cu113, pytorch-inf1, aarch64"
  exit 1;;
esac
