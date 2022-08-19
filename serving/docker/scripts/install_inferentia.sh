#!/usr/bin/env bash

set -e

# refer to: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-deploy/docker-example/Dockerfile-libmode.html#libmode-dockerfile
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gnupg2 curl software-properties-common

# install python 3.7 -- required by inferentia
add-apt-repository -y ppa:deadsnakes/ppa

# remove python 3.8
apt-get autoremove -y python3

echo "deb https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list
curl -L https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Installing Neuron Tools
apt-get update
apt-get install -y aws-neuron-tools python3.7 python3.7-distutils pciutils
ln -sf /usr/bin/python3.7 /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/bin/python

curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python3 -m pip install -U pip
rm get-pip.py

# Include torch-neuron
python3 -m pip install numpy
python3 -m pip install torch-neuron==1.11.0.* --extra-index-url=https://pip.repos.neuron.amazonaws.com
