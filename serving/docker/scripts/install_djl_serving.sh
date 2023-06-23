#!/usr/bin/env bash

set -ex

DJL_VERSION=$1
PYTORCH_JNI=$2

if [ -z "$PYTORCH_JNI" ]; then
  # install Java
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    openjdk-11-jdk-headless \
    curl \
    jq \
    unzip

  # install DJLServing
  curl https://publish.djl.ai/djl-serving/djl-serving_${DJL_VERSION}-1_all.deb -f -o djl-serving_all.deb
  dpkg -i djl-serving_all.deb
  rm djl-serving_all.deb
  cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/
  cp -r /usr/local/djl-serving-*/plugins /opt/djl/plugins
else
  if [[ ! "$DJL_VERSION" == *SNAPSHOT ]]; then
    djl-serving -i ai.djl.pytorch:pytorch-jni:${PYTORCH_JNI}-${DJL_VERSION}
    rm -rf /opt/djl/logs
  fi
fi

cd /usr/local/djl-serving-*/lib
rm netty*.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport-classes-epoll/4.1.94.Final/netty-transport-classes-epoll-4.1.94.Final.jar -f -o netty-transport-classes-epoll-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport-classes-kqueue/4.1.94.Final/netty-transport-classes-kqueue-4.1.94.Final.jar -f -o netty-transport-classes-kqueue-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport-native-epoll/4.1.94.Final/netty-transport-native-epoll-4.1.94.Final-linux-aarch_64.jar -f -o netty-transport-native-epoll-4.1.94.Final-linux-aarch_64.jar
curl https://repo1.maven.org/maven2/io/netty/netty-buffer/4.1.94.Final/netty-buffer-4.1.94.Final.jar -f -o netty-buffer-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-codec/4.1.94.Final/netty-codec-4.1.94.Final.jar  -f -o netty-codec-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-codec-http/4.1.94.Final/netty-codec-http-4.1.94.Final.jar -f -o netty-codec-http-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-common/4.1.94.Final/netty-common-4.1.94.Final.jar -f -o netty-common-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-handler/4.1.94.Final/netty-handler-4.1.94.Final.jar -f -o netty-handler-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-resolver/4.1.94.Final/netty-resolver-4.1.94.Final.jar -f -o netty-resolver-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport/4.1.94.Final/netty-transport-4.1.94.Final.jar  -f -o netty-transport-4.1.94.Final.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport-native-epoll/4.1.94.Final/netty-transport-native-epoll-4.1.94.Final-linux-x86_64.jar -f -o netty-transport-native-epoll-4.1.94.Final-linux-x86_64.jar
curl https://repo1.maven.org/maven2/io/netty/netty-transport-native-unix-common/4.1.94.Final/netty-transport-native-unix-common-4.1.94.Final.jar -f -o netty-transport-native-unix-common-4.1.94.Final.jar