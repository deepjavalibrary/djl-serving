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
FROM ubuntu:20.04 AS base
ARG djl_version=0.19.0~SNAPSHOT

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf
COPY config.properties /opt/djl/conf/
RUN scripts/install_djl_serving.sh $djl_version && rm -rf scripts

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"

FROM base AS cpu-full

COPY scripts scripts/
RUN scripts/install_python.sh && \
    scripts/install_djl_serving.sh $djl_version 1.11.0 && \
    djl-serving -i ai.djl.mxnet:mxnet-native-mkl:1.9.0:linux-x86_64 && \
    djl-serving -i ai.djl.pytorch:pytorch-native-cpu:1.11.0:linux-x86_64 && \
    djl-serving -i ai.djl.tensorflow:tensorflow-native-cpu:2.7.0:linux-x86_64 && \
    djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:0.18.0 && \
    djl-serving -i com.microsoft.onnxruntime:onnxruntime:1.12.1 && \
    scripts/patch_oss_dlc.sh python && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

