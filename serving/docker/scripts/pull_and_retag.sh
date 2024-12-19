#!/usr/bin/env bash
# for djl-serving/.github/workflows/nightly-docker-ecr-sync.yml

version=$1
repo=$2
images="cpu aarch64 cpu-full pytorch-inf2 pytorch-gpu lmi tensorrt-llm"

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
