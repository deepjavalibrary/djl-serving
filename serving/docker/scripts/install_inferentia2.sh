#!/usr/bin/env bash

set -ex

apt-get update -y && apt-get install -y --no-install-recommends \
  curl \
  git \
  gnupg2 \
  pciutils \
  udev

# Configure Linux for Neuron repository updates
. /etc/os-release
echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" >/etc/apt/sources.list.d/neuron.list
curl -L https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/releasecontent.html#inf2-packages
apt-get update -y && apt-get install -y aws-neuronx-collectives=2.20.11.0* \
    aws-neuronx-runtime-lib=2.20.11.0* \
    aws-neuronx-tools=2.17.0.0

# TODO: Remove this hack after aws-neuronx-dkms install no longer throws an error, this bypasses the `set -ex`
#       exit criteria. The package is installed and functional after running, just throws an error on install.
apt-get install -y aws-neuronx-dkms=2.15.9.0 || echo "Installed aws-neuronx-dkms with errors"

export PATH=/opt/aws/neuron/bin:$PATH
