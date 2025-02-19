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

base_tag="$version-$image"

if [[ "$mode" == "nightly" ]]; then
  base_tag="$base_tag-nightly"
fi
docker pull "$from_repo:$base_tag-$commit_sha"
docker tag "$from_repo:$base_tag-$commit_sha" "$to_repo:$base_tag"
docker push "$to_repo:$base_tag"
if [[ "$image" == "cpu" ]]; then
  if [[ "$mode" == "release" ]]; then
    docker tag "$from_repo:$base_tag-$commit_sha" "$to_repo:$version"
    docker push "$to_repo:$version"
    docker tag "$from_repo:$base_tag-$commit_sha" "$to_repo:latest"
    docker push "$to_repo:latest"
  elif [[ "$mode" == "nightly" ]]; then
    docker tag "$from_repo:$base_tag-$commit_sha" "$to_repo:$version-nightly"
    docker push "$to_repo:$version-nightly"
  fi
fi
