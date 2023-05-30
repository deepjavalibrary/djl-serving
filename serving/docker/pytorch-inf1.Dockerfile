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
FROM ubuntu:20.04
ARG djl_version=0.23.0~SNAPSHOT
ARG torch_version=1.12.1
EXPOSE 8080

# Sets up Path for Neuron tools
ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HUGGINGFACE_HUB_CACHE=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV NEURON_SDK_PATH=/usr/local/lib/python3.7/dist-packages/torch_neuron/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEURON_SDK_PATH
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.7/dist-packages/torch/lib
ENV PYTORCH_EXTRA_LIBRARY_PATH=$NEURON_SDK_PATH/libtorchneuron.so
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=1.12.1
ENV JAVA_OPTS="-Xmx1g -Xms1g -Xss2m -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError -Dai.djl.default_engine=PyTorch"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/
RUN scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} inf" > /opt/djl/bin/telemetry && \
    scripts/install_djl_serving.sh $djl_version ${torch_version} && \
    scripts/install_inferentia.sh && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh pytorch-inf1 && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.inf1="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
