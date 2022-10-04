#!/usr/bin/env bash

IMAGE_NAME=$1

apt-get update

if [[ "$IMAGE_NAME" == "deepspeed" ]]; then
  apt-get upgrade dpkg e2fsprogs libdpkg-perl libpcre2-8-0 libpcre3 openssl libsqlite3-0
fi
