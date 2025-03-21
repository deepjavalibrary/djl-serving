#!/usr/bin/env bash

set -ex

ARCH=$1

if [[ $ARCH == "aarch64" ]]; then
  curl https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-arm64.tar.gz -L -o s5cmd.tar.gz
else
  curl https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz -L -o s5cmd.tar.gz
fi

INSTALL_DIR="/opt/djl/bin"

mkdir -p "${INSTALL_DIR}"
tar -xvf s5cmd.tar.gz -C "${INSTALL_DIR}"
rm -rf s5cmd.tar.gz

export PATH="${INSTALL_DIR}:${PATH}"
echo "export PATH=${INSTALL_DIR}:\$PATH" >> ~/.bashrc
