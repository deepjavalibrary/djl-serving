#!/usr/bin/env bash
TORCH_NEURONX_VERSION=$1

set -ex

apt-get update -y && apt-get install -y --no-install-recommends \
  curl \
  git \
  gnupg2

# Configure Linux for Neuron repository updates
. /etc/os-release
echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" >/etc/apt/sources.list.d/neuron.list
curl -L https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

apt-get update -y && apt-get install -y linux-headers-$(uname -r) && apt-get install -y aws-neuronx-dkms=2.* \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-tools=2.*

export PATH=/opt/aws/neuron/bin:$PATH

python3 -m pip install numpy awscli
python3 -m pip install neuronx-cc==2.* torch_neuronx==${TORCH_NEURONX_VERSION} --extra-index-url=https://pip.repos.neuron.amazonaws.com
