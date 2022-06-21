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
ARG version=11.3.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.17.0
ARG torch_version=1.11.0
ARG deepspeed_version=0.6.5
ARG transformers_version=4.19.2

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps
COPY config.properties /opt/djl/conf/
RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_python.sh

### Deep Speed installations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev && \
    pip3 install torch==${torch_version} --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip3 install deepspeed==${deepspeed_version} transformers==${transformers_version} triton==1.0.0 mpi4py

RUN rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
