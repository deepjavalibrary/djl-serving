#!/usr/bin/env bash
# for docker_publish.yml
version=$1
repo=$2
mode=$3
images="cpu aarch64 cpu-full pytorch-inf2 pytorch-gpu lmi tensor rt-llm"

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
  echo docker pull $temprepo:$image-$mode-$GITHUB_RUN_ID
  echo docker tag $temprepo:$image-$mode-$GITHUB_RUN_ID $repo:$tag
  echo docker push $repo:$tag
done