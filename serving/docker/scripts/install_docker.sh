#!/usr/bin/env bash

if [ -x "$(command -v docker)" ]; then
    echo "Docker is installed! skipped..."
    exit 0
fi

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo chmod 666 /var/run/docker.sock
