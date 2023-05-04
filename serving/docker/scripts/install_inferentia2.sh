#!/usr/bin/env bash

set -ex

apt-get update -y && apt-get install -y --no-install-recommends \
  curl \
  git \
  gnupg2 \
  pciutils \
  python3-pip \
  g++

ln -sf /usr/bin/python3 /usr/bin/python

# Configure Linux for Neuron repository updates
. /etc/os-release
echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" >/etc/apt/sources.list.d/neuron.list
curl -L https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/releasecontent.html#inf2-packages
apt-get update -y && apt-get install -y linux-headers-$(uname -r) && apt-get install -y \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-tools=2.*

python3 -m pip --no-cache-dir install -U pip
# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx torchvision

export PATH=/opt/aws/neuron/bin:$PATH
