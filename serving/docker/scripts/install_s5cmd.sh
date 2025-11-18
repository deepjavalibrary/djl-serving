#!/usr/bin/env bash

set -ex

ARCH=$1

# Install jq if not available (for lmi container)
if ! command -v jq &> /dev/null; then
    apt-get update && apt-get install -y jq
fi

# Retrieve latest patched version
GO_MAJOR_MINOR="1.25"
GO_VERSION=$(curl -s https://go.dev/dl/?mode=json | jq -r ".[].version" | grep "^go${GO_MAJOR_MINOR}" | head -1 | sed 's/go//')
echo "Using Go version: ${GO_VERSION} (latest in ${GO_MAJOR_MINOR}.x series)"

if [[ $ARCH == "aarch64" ]]; then
  GO_ARCH="arm64"
else
  GO_ARCH="amd64"
fi

curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz" | tar -xz -C /tmp
export PATH="/tmp/go/bin:${PATH}"
export GOPATH="/tmp/gopath"
export GOCACHE="/tmp/gocache"

# Download s5cmd release source
S5CMD_VERSION="v2.3.0"
echo "Building s5cmd ${S5CMD_VERSION}"
curl -fsSL "https://github.com/peak/s5cmd/archive/refs/tags/${S5CMD_VERSION}.tar.gz" | tar -xz -C /tmp
mv /tmp/s5cmd-${S5CMD_VERSION#v} /tmp/s5cmd
cd /tmp/s5cmd
go build -ldflags "-X github.com/peak/s5cmd/v2/version.Version=${S5CMD_VERSION}" -o s5cmd .

# Install s5cmd
INSTALL_DIR="/opt/djl/bin"
mkdir -p "${INSTALL_DIR}"
mv s5cmd "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/s5cmd"

rm -rf /tmp/go /tmp/gopath /tmp/gocache /tmp/s5cmd

export PATH="${INSTALL_DIR}:${PATH}"
echo "export PATH=${INSTALL_DIR}:\$PATH" >>~/.bashrc
