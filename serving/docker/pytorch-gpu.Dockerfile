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
ARG version=12.4.1-devel-ubuntu22.04

FROM nvidia/cuda:$version as base

ARG djl_version
ARG djl_serving_version
ARG cuda_version=cu124
ARG torch_version=2.5.1
ARG torch_vision_version=0.20.1
ARG onnx_version=1.19.0
ARG python_version=3.10
ARG numpy_version=1.26.4
ARG pydantic_version=2.8.2
ARG djl_converter_wheel="https://publish.djl.ai/djl_converter/djl_converter-0.31.0-py3-none-any.whl"

RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
# ENV NO_OMP_NUM_THREADS=true
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HF_HOME=/tmp/.cache/huggingface
# set cudnn9 library path
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=${torch_version}
ENV PYTORCH_FLAVOR=cu124-precxx11
# TODO: remove TORCH_CUDNN_V8_API_DISABLED once PyTorch bug is fixed
ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError -Dai.djl.default_engine=PyTorch"
ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

COPY scripts scripts/
SHELL ["/bin/bash", "-c"]
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version ${torch_version} && \
    djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:$djl_version && \
    djl-serving -i com.microsoft.onnxruntime:onnxruntime_gpu:$onnx_version && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_s5cmd.sh x64 && \
    pip3 install peft pydantic==${pydantic_version} ${djl_converter_wheel} hf-transfer && \
    pip3 install numpy==${numpy_version} && pip3 install torch==${torch_version} torchvision==${torch_vision_version} --extra-index-url https://download.pytorch.org/whl/cu124 && \
    scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh pytorch-gpu && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_serving_version} pytorchgpu" > /opt/djl/bin/telemetry && \
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
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.pytorch-cu124="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL cuda-version=$cuda_version
LABEL torch-version=$torch_version
# To use the 535 CUDA driver
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
