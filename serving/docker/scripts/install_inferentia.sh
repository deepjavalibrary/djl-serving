#!/usr/bin/env bash
# refer to: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-deploy/docker-example/Dockerfile-libmode.html#libmode-dockerfile
apt-get install -y --no-install-recommends gnupg2 wget

echo "deb https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -


# Installing Neuron Tools
apt-get update -y
apt-get install -y aws-neuron-tools

# install python 3.7 -- required by inferentia
add-apt-repository ppa:deadsnakes/ppa
apt-get install python3.7
wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py

# Include torch-neuron
pip3 install numpy \
    && pip3 install torch-neuron==1.11.0.* \
      --extra-index-url=https://pip.repos.neuron.amazonaws.com
