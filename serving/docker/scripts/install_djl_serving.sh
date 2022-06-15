#!/usr/bin/env bash
# install Java
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
fakeroot \
openjdk-11-jdk-headless \
curl
# install DJLServing
dpkg -i djl-serving_all.deb
rm djl-serving_all.deb
cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/
