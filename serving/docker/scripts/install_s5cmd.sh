#!/usr/bin/env bash

set -ex

ARCH=$1

if [[ $ARCH == "aarch64" ]]; then
  curl https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-arm64.tar.gz -L -o s5cmd.tar.gz
else
  curl https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz -L -o s5cmd.tar.gz
fi

mkdir -p /opt/djl/bin
tar -xvf s5cmd.tar.gz -C /opt/djl/bin
rm -rf s5cmd.tar.gz
