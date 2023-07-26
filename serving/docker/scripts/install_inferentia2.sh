#!/usr/bin/env bash

set -ex

apt-get update -y && apt-get install -y --no-install-recommends \
  curl \
  git \
  gnupg2 \
  pciutils

# Configure Linux for Neuron repository updates
. /etc/os-release
echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" >/etc/apt/sources.list.d/neuron.list
curl -L https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/releasecontent.html#inf2-packages
apt-get update -y && apt-get install -y linux-headers-$(uname -r) && apt-get install -y aws-neuronx-dkms=2.11.* \
    aws-neuronx-collectives=2.15.* \
    aws-neuronx-runtime-lib=2.15.* \
    aws-neuronx-tools=2.12.*

export PATH=/opt/aws/neuron/bin:$PATH
