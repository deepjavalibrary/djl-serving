#!/usr/bin/env bash

set -ex

DJL_VERSION=$1
DJL_SERVING_VERSION=$2
PYTORCH_JNI=$3

if [ -z "$PYTORCH_JNI" ]; then
  # install Java
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    wget \
    gpg \
    curl \
    jq \
    unzip \
    ca-certificates \
    fontconfig \
    vim
# add corretto https://docs.aws.amazon.com/corretto/latest/corretto-17-ug/generic-linux-install.html
  wget -O - https://apt.corretto.aws/corretto.key | gpg --dearmor -o /usr/share/keyrings/corretto-keyring.gpg &&
    echo "deb [signed-by=/usr/share/keyrings/corretto-keyring.gpg] https://apt.corretto.aws stable main" | tee /etc/apt/sources.list.d/corretto.list
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    java-17-amazon-corretto-jdk

  # install DJLServing package 
  if [ ! -f djl-serving_all.deb ]; then
    curl "https://publish.djl.ai/djl-serving/djl-serving_${DJL_SERVING_VERSION//-/\~}-1_all.deb" -f -o djl-serving_all.deb
  fi
  dpkg -i djl-serving_all.deb
  rm djl-serving_all.deb

  mkdir -p /opt/djl/plugins
else
  djl-serving -i "ai.djl.pytorch:pytorch-jni:${PYTORCH_JNI}-${DJL_VERSION}"
  rm -rf /opt/djl/logs
fi
