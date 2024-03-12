#!/usr/bin/env bash
# for GitHub Action use only
arch=$1
version=$2

if [[ -z $version || "$version" == "nightly" ]]; then
  image="$arch-nightly"
elif [[ "$version" == "temp"  ]]; then
  repo="185921645874.dkr.ecr.us-east-1.amazonaws.com/djl-ci-temp"
  image="$repo:$arch-$GITHUB_SHA"
  echo "DJLSERVING_TEMP_REPO=$repo" >> $GITHUB_ENV
else
  if [[ "$arch" == "cpu" ]]; then
    image=$version
  else
    image="$version-$arch"
  fi
fi

echo "DJLSERVING_DOCKER=$image" >> $GITHUB_ENV
