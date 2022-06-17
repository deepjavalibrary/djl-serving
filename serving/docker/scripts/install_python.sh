#!/usr/bin/env bash
# Ubuntu 20.04 ships python3.8 by default
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip
pip3 --no-cache-dir install --upgrade pip
ln -s /usr/bin/python3 /usr/bin/python
