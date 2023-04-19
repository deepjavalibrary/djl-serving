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
ARG version=11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.22.1~SNAPSHOT
ARG python_version=3.9
ARG torch_version=2.0.0
ARG xformers_version=0.0.18
ARG accelerate_version=0.18.0
ARG deepspeed_wheel="https://publish.djl.ai/deepspeed/deepspeed-0.8.3-py2.py3-none-any.whl"
ARG transformers_version=4.27.4
ARG diffusers_version=0.14.0

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HUGGINGFACE_HUB_CACHE=/tmp
ENV TRANSFORMERS_CACHE=/tmp

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/djl/partition
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition

RUN apt-get update && \
    scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} deepspeed" > /opt/djl/bin/telemetry && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_s5cmd.sh x64 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev && \
    pip3 install torch==${torch_version} --extra-index-url https://download.pytorch.org/whl/cu118 \
    ${deepspeed_wheel} transformers==${transformers_version} \
    triton==2.0.0.dev20221202 mpi4py sentencepiece accelerate==${accelerate_version} bitsandbytes && \
    pip3 install diffusers[torch]==${diffusers_version} xformers==${xformers_version} && \
    scripts/install_aitemplate.sh && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh deepspeed && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.deepspeed="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
