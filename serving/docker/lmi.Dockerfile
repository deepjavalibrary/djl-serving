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
ARG version=12.8.1-devel-ubuntu24.04
FROM nvidia/cuda:$version
ARG cuda_version=cu128
ARG djl_version
ARG djl_serving_version
ARG python_version=3.11
ARG djl_torch_version=2.5.1
ARG djl_onnx_version=1.20.0

# djl converter wheel for text-embedding use case
ARG djl_converter_wheel="https://publish.djl.ai/djl_converter/djl_converter-${djl_version//-*/}-py3-none-any.whl"

EXPOSE 8080

COPY dockerd-entrypoint-with-cuda-compat.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
# ENV NO_OMP_NUM_THREADS=true
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError -Dai.djl.util.cuda.fork=true"
ENV MODEL_SERVER_HOME=/opt/djl
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV DJL_CACHE_DIR=/tmp/.djl.ai
# set cudnn9 library path
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python${python_version}/dist-packages/nvidia/cudnn/lib/"
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python${python_version}/dist-packages/torch/lib
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=${djl_torch_version}
ENV PYTORCH_FLAVOR=cu124-precxx11
ENV VLLM_NO_USAGE_STATS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
# 0.6.2 is the last version that contains legacy support for beam search
# TODO: update beam search logic and implementation in handlers
ENV VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1
ENV HF_HOME=/tmp/.cache/huggingface
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV BITSANDBYTES_NOWELCOME=1
ENV USE_AICCL_BACKEND=true
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV SAFETENSORS_FAST_GPU=1
ENV TORCH_NCCL_BLOCKING_WAIT=0
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV SERVING_FEATURES=vllm
ENV DEBIAN_FRONTEND=noninteractive
# Making s5cmd discoverable
ENV PATH="/opt/djl/bin:${PATH}"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN chmod -R +x scripts
RUN mkdir -p /opt/djl/conf \
    && mkdir -p /opt/djl/deps \
    && mkdir -p /opt/djl/partition \
    && mkdir -p /opt/ml/model \
    && mkdir -p /opt/djl/bin \
    && echo "${djl_serving_version} lmi" > /opt/djl/bin/telemetry
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition
COPY scripts/telemetry.sh /opt/djl/bin

RUN apt-get update && apt-get install -yq libaio-dev libopenmpi-dev g++ unzip cuda-compat-12-8 \
    && scripts/install_openssh.sh \
    && scripts/install_python.sh ${python_version} \
    && scripts/install_s5cmd.sh x64 \
    && pip3 cache purge \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN scripts/patch_oss_dlc.sh python \
    && scripts/security_patch.sh lmi \
    && useradd -m -d /home/djl djl \
    && chown -R djl:djl /opt/djl \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY lmi-container-requirements-common.txt ./requirements-common.txt
COPY requirements-vllm.txt ./requirements-vllm.txt
RUN pip3 install -r requirements-common.txt \
    && pip3 install ${djl_converter_wheel} --no-deps \
    && scripts/create_virtual_env.sh /opt/djl/vllm_venv requirements-vllm.txt

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

RUN scripts/install_djl_serving.sh $djl_version $djl_serving_version ${djl_torch_version} \
    && djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:$djl_version \
    && djl-serving -i com.microsoft.onnxruntime:onnxruntime_gpu:$djl_onnx_version

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.lmi="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.lmi="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver, CUDA 12.8 can work on this one too
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
