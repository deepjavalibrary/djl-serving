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
ARG version=12.1.1-cudnn8-devel-ubuntu22.04

FROM nvidia/cuda:$version as base

ARG djl_version=0.27.0~SNAPSHOT
ARG cuda_version=cu121
ARG torch_version=2.1.1
ARG torch_vision_version=0.16.1
ARG python_version=3.10

RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# ENV NO_OMP_NUM_THREADS=true
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=${torch_version}
ENV PYTORCH_FLAVOR=cu121-precxx11
# TODO: remove TORCH_CUDNN_V8_API_DISABLED once PyTorch bug is fixed
ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError -Dai.djl.default_engine=PyTorch"
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

COPY scripts scripts/
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} pytorchgpu" > /opt/djl/bin/telemetry && \
    scripts/install_djl_serving.sh $djl_version ${torch_version} && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_s5cmd.sh x64 && \
    pip3 install numpy && pip3 install torch==${torch_version} torchvision==${torch_vision_version} --extra-index-url https://download.pytorch.org/whl/cu121 && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh pytorch-gpu && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*


EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.pytorch-gpu="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-26-0.pytorch-cu121="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL cuda-version=$cuda_version
LABEL torch-version=$torch_version
# To use the 535 CUDA driver
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
