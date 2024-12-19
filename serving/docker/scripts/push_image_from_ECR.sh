#!/usr/bin/env bash
# for docker_publish.yml

set -euo pipefail
# Validate required arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <version> <to_repo> <mode> [commit_sha]" >&2
    exit 1
fi
# Validate required environment variables
if [ -z "$AWS_TMP_ECR_REPO" ]; then
    echo "ERROR: AWS_TMP_ECR_REPO environment variable is not set" >&2
    exit 1
fi

version=$1
to_repo=$2
mode=$3
commit_sha=${4:-$GITHUB_SHA}  # Use parameter expansion for default value

images=(cpu aarch64 cpu-full pytorch-inf2 pytorch-gpu lmi tensorrt-llm)

from_repo=$AWS_TMP_ECR_REPO

set -x
for image in "${images[@]}"; do

  if [[ "$mode" == "release" ]]; then
    if [[ "$image" == "cpu" ]]; then
      tag=$version
    else
      tag="$version-$image"
    fi
  fi

  if [[ "$mode" == "nightly" ]]; then
    tag="$image-nightly"
  fi
  docker pull $from_repo:$image-$mode-$commit_sha
  echo docker tag $from_repo:$image-$mode-$commit_sha $to_repo:$tag
  echo docker push $to_repo:$tag
done