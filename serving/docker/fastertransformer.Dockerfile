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
ARG version=11.6.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.21.0~SNAPSHOT
ARG ft_version="release/v5.2.1_tag"
ARG torch_wheel="https://aws-pytorch-unified-cicd-binaries.s3.us-west-2.amazonaws.com/r1.12.1_ec2/20221208-234008/d3dae914337cde7e182d28544aed5efce29255c4/torch-1.12.1%2Bcu116-cp38-cp38-linux_x86_64.whl"
ARG ompi_version=4.1.4

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
COPY config.properties /opt/djl/conf/config.properties

# Install all dependencies
RUN apt-get update && apt-get install -y wget git && \
    mkdir ompi && cd ompi && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${ompi_version}.tar.gz | tar xzf - && \
    cd openmpi-${ompi_version} && \
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/openmpi-${ompi_version} --with-cuda && \
    make -j${nproc} -s install && \
    ln -s /usr/local/openmpi-${ompi_version} /usr/local/mpi && \
    cd ../../ && rm -rf ompi && \
    scripts/install_python.sh && \
    pip3 install ${torch_wheel} && \
    pip3 install cmake && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

ENV PATH=/usr/local/mpi/bin:${PATH} LD_LIBRARY_PATH=/usr/local/mpi/lib:${LD_LIBRARY_PATH}

# Supporting build g4,g5,p3,p4
RUN git clone https://github.com/NVIDIA/FasterTransformer.git -b ${ft_version} \
    && mkdir -p FasterTransformer/build \
    && cd FasterTransformer/build \
    && git submodule init && git submodule update \
    && cmake -DCMAKE_BUILD_TYPE=Release -DSM=70,75,80,86 -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON .. \
    && make -j${nproc} install \
    && rm -rf lib/*TritonBackend.so \
    && cp lib/*.so /usr/local/backends/fastertransformer/ \
    && mkdir -p /usr/local/backends/fastertransformer/bin \
    && cp -r bin/*_gemm /usr/local/backends/fastertransformer/bin/ \
    && cd ../../ && rm -rf FasterTransformer

# TODO: Add DLC patching
RUN apt-get update && \
    scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} fastertransformer" > /opt/djl/bin/telemetry && \
    scripts/install_s5cmd.sh x64 && \
    scripts/patch_oss_dlc.sh python && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.fastertransformer="true"
