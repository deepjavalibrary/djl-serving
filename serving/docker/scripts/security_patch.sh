#!/usr/bin/env bash

IMAGE_NAME=$1

apt-get update

if [[ "$IMAGE_NAME" == "deepspeed" ]]; then
  apt-get upgrade -y dpkg e2fsprogs libdpkg-perl libpcre2-8-0 libpcre3 openssl libsqlite3-0 libdbus-1-3 curl
elif [[ "$IMAGE_NAME" == "pytorch-cu113" ]]; then
  apt-get upgrade -y dpkg e2fsprogs libdpkg-perl libpcre2-8-0 libpcre3 openssl libsqlite3-0 libsepol1 libdbus-1-3 curl
elif [[ "$IMAGE_NAME" == "cpu" ]]; then
  apt-get upgrade -y libpcre2-8-0 libdbus-1-3 curl
fi
