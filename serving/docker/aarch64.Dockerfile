# -*- mode: dockerfile -*-
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
FROM arm64v8/ubuntu:20.04
ARG djl_version=0.19.0~SNAPSHOT
ARG torch_version=1.12.1

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV OMP_NUM_THREADS=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError -Dai.djl.default_engine=PyTorch"
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps
COPY config.properties /opt/djl/conf/

RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_djl_serving.sh $djl_version $torch_version && \
    djl-serving -i ai.djl.pytorch:pytorch-native-cpu-precxx11:$torch_version:linux-aarch64 && \
    rm -f /usr/local/djl-serving-*/lib/mxnet-* && \
    rm -f /usr/local/djl-serving-*/lib/tensorflow-* && \
    rm -f /usr/local/djl-serving-*/lib/tensorrt-* && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
