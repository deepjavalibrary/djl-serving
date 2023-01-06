#!/usr/bin/env bash

IMAGE_NAME=$1

apt-get update

if [[ "$IMAGE_NAME" == "deepspeed" ]]; then
  apt-get upgrade -y dpkg e2fsprogs libdpkg-perl libpcre2-8-0 libpcre3 openssl libsqlite3-0 libdbus-1-3 curl libksba8
elif [[ "$IMAGE_NAME" == "pytorch-cu117" ]]; then
  apt-get upgrade -y dpkg e2fsprogs libdpkg-perl libpcre2-8-0 libpcre3 openssl libsqlite3-0 libsepol1 libdbus-1-3 curl
elif [[ "$IMAGE_NAME" == "cpu" ]] ||  [[ "$IMAGE_NAME" == "pytorch-inf1" ]]; then
  apt-get upgrade -y libpcre2-8-0 libdbus-1-3 curl
fi
