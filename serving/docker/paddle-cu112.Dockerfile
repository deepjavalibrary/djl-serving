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
FROM paddlepaddle/paddle:2.3.2-gpu-cuda11.2-cudnn8
ARG djl_version=0.21.0~SNAPSHOT

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf
COPY config.properties /opt/djl/conf/
RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_s5cmd.sh x64 && \
    pip3 install --upgrade pip && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} paddlegpu" > /opt/djl/bin/telemetry && \
    scripts/security_patch.sh pytorch-cu117 && \
    scripts/patch_oss_dlc.sh python && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    rm -rf scripts

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:-UseContainerSupport -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.paddle-cu112="true"
