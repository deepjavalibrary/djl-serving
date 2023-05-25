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
ARG version=11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.19.0~SNAPSHOT
ARG torch_wheel="https://aws-pytorch-unified-cicd-binaries.s3.us-west-2.amazonaws.com/r1.12.1_ec2/20221208-233710/d3dae914337cde7e182d28544aed5efce29255c4/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl"
ARG deepspeed_version=0.7.3
ARG accelerate_version=0.13.2
ARG transformers_version=4.22.1
ARG bitsandbytes_version=0.38.1

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps
COPY deepspeed.config.properties /opt/djl/conf/config.properties
### Install DJLServing and update latest python engine, TODO: stop updatign python engine once it stable
RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_python.sh && \
    cd /usr/local/djl-serving-*/lib/ && \
    rm -rf python*.jar && \
    curl -f -O https://publish.djl.ai/djl-serving/python/python-${djl_version}.jar

### Deep Speed installations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev && \
    pip3 install ${torch_wheel} && \
    pip3 install deepspeed==${deepspeed_version} transformers==${transformers_version} && \
    pip3 install triton==1.0.0 mpi4py sentencepiece accelerate==${accelerate_version} bitsandbytes==${bitsandbytes_version} && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh deepspeed && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.deepspeed="true"
