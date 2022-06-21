#!/usr/bin/env bash

set -e

DJL_VERSION=$1

# install Java
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  fakeroot \
  openjdk-11-jdk-headless \
  curl

# install DJLServing
curl https://publish.djl.ai/djl-serving/djl-serving_${DJL_VERSION}-1_all.deb -o djl-serving_all.deb
dpkg -i djl-serving_all.deb
rm djl-serving_all.deb
cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/
