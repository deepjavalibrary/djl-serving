#!/usr/bin/env bash

set -ex

ARCH=$1

# Download custom s5cmd binary built with Go 1.25.4
if [[ $ARCH == "aarch64" ]]; then
  curl -f https://publish.djl.ai/s5cmd/go1.25.4/s5cmd-linux-arm64 -L -o s5cmd
else
  curl -f https://publish.djl.ai/s5cmd/go1.25.4/s5cmd-linux-amd64 -L -o s5cmd
fi

INSTALL_DIR="/opt/djl/bin"

mkdir -p "${INSTALL_DIR}"
mv s5cmd "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/s5cmd"

export PATH="${INSTALL_DIR}:${PATH}"
echo "export PATH=${INSTALL_DIR}:\$PATH" >>~/.bashrc
