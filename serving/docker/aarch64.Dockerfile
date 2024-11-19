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
FROM arm64v8/ubuntu:22.04
ARG djl_version
ARG djl_serving_version
ARG torch_version=2.5.1

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError -Dai.djl.default_engine=PyTorch"
ENV MODEL_SERVER_HOME=/opt/djl
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV DNNL_DEFAULT_FPMATH_MODE=BF16
ENV LRU_CACHE_CAPACITY=1024

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

RUN scripts/install_djl_serving.sh $djl_version $djl_serving_version && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version $torch_version && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version $torch_version && \
    scripts/install_s5cmd.sh aarch64 && \
    djl-serving -i ai.djl.pytorch:pytorch-native-cpu-precxx11:$torch_version:linux-aarch64 && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_serving_version} aarch" > /opt/djl/bin/telemetry && \
    rm -f /usr/local/djl-serving-*/lib/mxnet-* && \
    rm -f /usr/local/djl-serving-*/lib/tensorflow-* && \
    rm -f /usr/local/djl-serving-*/lib/tensorrt-* && \
    rm -rf /opt/djl/logs && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.aarch64="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.aarch64="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL torch-version=$torch_version
