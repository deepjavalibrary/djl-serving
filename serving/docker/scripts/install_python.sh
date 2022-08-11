#!/usr/bin/env bash

set -e

# Ubuntu 20.04 ships python3.8 by default
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip
python3 -m pip --no-cache-dir install -U pip --upgrade
python3 -m pip --no-cache-dir install -U numpy
ln -s /usr/bin/python3 /usr/bin/python
