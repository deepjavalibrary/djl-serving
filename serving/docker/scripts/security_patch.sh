#!/usr/bin/env bash

IMAGE_NAME=$1

apt-get update

if [[ "$IMAGE_NAME" == "deepspeed" ]] || \
   [[ "$IMAGE_NAME" == "pytorch-gpu" ]]; then
  apt-get upgrade -y dpkg openssl curl libssl3
elif [[ "$IMAGE_NAME" == "cpu" ]]; then
  apt-get upgrade -y libpcre2-8-0 libdbus-1-3 curl
elif [[ "$IMAGE_NAME" == "trtllm" ]]; then
  apt-get upgrade -y libssl3
fi
