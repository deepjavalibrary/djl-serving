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
image=$4
commit_sha=${5:-$GITHUB_SHA}  # Use parameter expansion for default value


from_repo=$AWS_TMP_ECR_REPO

set -x

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
docker tag $from_repo:$image-$mode-$commit_sha $to_repo:$tag
docker push $to_repo:$tag
