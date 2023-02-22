#!/usr/bin/env bash

version=$1
repo=$2
images="cpu aarch64 cpu-full pytorch-inf1 pytorch-cu117 deepspeed transformers fastertransformer"

for image in $images; do
    if [[ ! "$version" == "nightly" ]]; then
      if [[ "$image" == "cpu" ]]; then
        image=$version
      else
        image="$version-$image"
      fi
    else
      image="$image-$version"
    fi
    docker pull deepjavalibrary/djl-serving:$image
    docker tag deepjavalibrary/djl-serving:$image $repo/djl-serving:$image
    docker push $repo/djl-serving:$image
done
