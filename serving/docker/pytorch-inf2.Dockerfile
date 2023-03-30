# -*- mode: dockerfile -*-
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
ARG djl_version=0.21.0~SNAPSHOT
ARG torch_version=1.13.1
ARG python_version=3.8
ARG torch_neuronx_version=1.13.0.1.4.0
ARG tnx_version=v2.8.0
EXPOSE 8080

# Sets up Path for Neuron tools
ENV PATH="/opt/aws/neuron/bin:${PATH}"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV MODEL_SERVER_HOME=/opt/djl
ENV NEURON_SDK_PATH=/usr/local/lib/python3.8/dist-packages/torch_neuronx/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEURON_SDK_PATH
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib
ENV PYTORCH_EXTRA_LIBRARY_PATH=$NEURON_SDK_PATH/libtorchneuron.so
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=1.13.1
ENV JAVA_OPTS="-Xmx1g -Xms1g -Xss2m -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps
COPY config.properties /opt/djl/conf/
RUN mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} inf2" > /opt/djl/bin/telemetry && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_djl_serving.sh $djl_version && \
    scripts/install_inferentia2.sh $torch_neuronx_version && \
    pip install "git+https://github.com/aws-neuron/transformers-neuronx.git@${tnx_version}" && \
    scripts/install_s5cmd.sh x64 && \
    scripts/patch_oss_dlc.sh python && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.inf2="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
