#!/usr/bin/env bash

set -ex

# Ubuntu 20.04 ships python3.8 by default
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip
python3 -m pip --no-cache-dir install -U pip
python3 -m pip --no-cache-dir install -U numpy awscli
ln -sf /usr/bin/python3 /usr/bin/python
