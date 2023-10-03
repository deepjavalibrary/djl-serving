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
ARG version=11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.24.0~SNAPSHOT
ARG python_version=3.9
ARG ft_version="llama"
ARG triton_version="r23.04"
ARG torch_wheel="https://aws-pytorch-unified-cicd-binaries.s3.us-west-2.amazonaws.com/r1.13.1_ec2/20221219-193736/54406b8eed7fbd61be629cb06229dfb7b6b2954e/torch-1.13.1%2Bcu117-cp39-cp39-linux_x86_64.whl"
ARG ft_wheel="https://publish.djl.ai/fastertransformer/fastertransformer-0.24.0-py3-none-any.whl"
ARG tb_wheel="https://publish.djl.ai/tritonserver/r23.04/tritontoolkit-23.4-py3-none-any.whl"
ARG peft_wheel="https://publish.djl.ai/peft/peft-0.5.0alpha-py3-none-any.whl"
ARG seq_scheduler_wheel="https://publish.djl.ai/seq_scheduler/seq_scheduler-0.1.0-py3-none-any.whl"
ARG ompi_version=4.1.4
ARG protobuf_version=3.20.3
ARG transformers_version=4.33.2
ARG accelerate_version=0.23.0
ARG bitsandbytes_version=0.41.1
ARG optimum_version=1.13.2
ARG auto_gptq_version=0.4.2

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# ENV NO_OMP_NUM_THREADS=true
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV BITSANDBYTES_NOWELCOME=1

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/djl/partition && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition

# Install all dependencies
RUN apt-get update && apt-get install -y wget git libnuma-dev zlib1g-dev rapidjson-dev && \
    mkdir ompi && cd ompi && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${ompi_version}.tar.gz | tar xzf - && \
    cd openmpi-${ompi_version} && \
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/openmpi-${ompi_version} --with-cuda && \
    make -j${nproc} -s install && \
    ln -s /usr/local/openmpi-${ompi_version} /usr/local/mpi && \
    cd ../../ && rm -rf ompi && \
    scripts/install_python.sh ${python_version} && \
    pip3 install ${torch_wheel} ${ft_wheel} ${tb_wheel} ${peft_wheel} ${seq_scheduler_wheel} safetensors protobuf==${protobuf_version} && \
    pip3 install transformers==${transformers_version} accelerate==${accelerate_version} \
    bitsandbytes==${bitsandbytes_version} optimum==${optimum_version} auto-gptq==${auto_gptq_version} \
    scipy einops && \
    pip3 install cmake sentencepiece bfloat16 tiktoken && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

ENV PATH=/usr/local/mpi/bin:${PATH} LD_LIBRARY_PATH=/usr/local/mpi/lib:${LD_LIBRARY_PATH}

# Install fastertransformer and triton
RUN mkdir -p /opt/tritonserver/backends/fastertransformer && mkdir -p /opt/tritonserver/lib && \
    curl -o /opt/tritonserver/lib/libtritonserver.so https://publish.djl.ai/tritonserver/${triton_version}/libtritonserver.so && \
    curl -o /opt/tritonserver/backends/fastertransformer/libth_transformer.so https://publish.djl.ai/fastertransformer/${ft_version}/libth_transformer.so && \
    curl -o /opt/tritonserver/backends/fastertransformer/libtransformer-shared.so https://publish.djl.ai/fastertransformer/${ft_version}/libtransformer-shared.so && \
    curl -o /opt/tritonserver/backends/fastertransformer/libtriton_fastertransformer.so https://publish.djl.ai/fastertransformer/${ft_version}/libtriton_fastertransformer.so && \
    curl -o /root/FASTERTRANSFORMER_LICENSE https://raw.githubusercontent.com/NVIDIA/FasterTransformer/main/LICENSE

ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:${LD_LIBRARY_PATH}

RUN apt-get update && \
    scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} fastertransformer" > /opt/djl/bin/telemetry && \
    scripts/install_s5cmd.sh x64 && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh fastertransformer && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.fastertransformer="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-24-0.fastertransformer="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
