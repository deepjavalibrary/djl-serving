#!/usr/bin/env bash

set -ex

# Install OpenSSH for MPI to communicate between containers, allow OpenSSH to talk to containers without asking for confirmation
apt-get update
apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends
apt-get install -y --no-install-recommends openssh-client openssh-server
mkdir -p /var/run/sshd
cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking >/etc/ssh/ssh_config.new
echo "    StrictHostKeyChecking no" >>/etc/ssh/ssh_config.new
mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
rm -rf /var/lib/apt/lists/*
apt-get clean

# Configure OpenSSH so that nodes can communicate with each other
mkdir -p /var/run/sshd
mkdir -p /root/.ssh
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

printf \"Port 2022\n\" >>/etc/ssh/sshd_config
printf \"Port 2022\n\" >>/root/.ssh/config
