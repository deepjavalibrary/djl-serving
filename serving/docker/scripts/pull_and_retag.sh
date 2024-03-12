#!/usr/bin/env bash

version=$1
repo=$2
mode=$3
images="cpu aarch64 cpu-full pytorch-inf2 pytorch-gpu deepspeed tensorrt-llm"

temprepo="185921645874.dkr.ecr.us-east-1.amazonaws.com/djl-ci-temp"

for image in $images; do
    if [[ ! "$mode" == "nightly" ]]; then
      if [[ "$image" == "cpu" ]]; then
        tag=$version
      else
        tag="$version-$image"
      fi
    else
      tag="$image-nightly"
    fi
    docker pull $temprepo:$image-$GITHUB_SHA
    docker tag $temprepo:$image-$GITHUB_SHA $repo:$tag
    docker push $repo:$tag
done
