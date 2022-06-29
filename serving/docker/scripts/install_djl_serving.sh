#!/usr/bin/env bash

set -e

DJL_VERSION=$1
PYTORCH_JNI=$2

if [ -z "$PYTORCH_JNI" ]; then
  # install Java
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    openjdk-11-jdk-headless \
    curl

  # install DJLServing
  curl https://publish.djl.ai/djl-serving/djl-serving_${DJL_VERSION}-1_all.deb -o djl-serving_all.deb
  dpkg -i djl-serving_all.deb
  rm djl-serving_all.deb
  cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/
else
  if [[ ! "$DJL_VERSION" == *SNAPSHOT ]]; then
    djl-serving -i ai.djl.pytorch:pytorch-jni:${PYTORCH_JNI}-${DJL_VERSION}
  fi
fi
