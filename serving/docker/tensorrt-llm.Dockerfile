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
ARG version=12.2.2-runtime-ubuntu22.04
FROM nvidia/cuda:$version
ARG python_version=3.10
ARG TRT_LLM_VERSION=release/0.5.0
ARG TORCH_VERSION=2.1.0
ARG djl_version=0.24.0~SNAPSHOT

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
# TRT ENV
ENV TRT_ROOT=/usr/local/tensorrt
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/djl/partition && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

# Install OpenMPI and other deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget unzip openmpi-bin libopenmpi-dev libffi-dev git-lfs rapidjson-dev && \
    scripts/install_python.sh ${python_version} && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install torch==${TORCH_VERSION} && \
    pip3 cache purge

# Install TRT
RUN scripts/install_tensorrt.sh && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# download dependencies
RUN pip install https://publish.djl.ai/tensorrt-llm/0.5.0/tensorrt_llm-0.5.0-py3-none-any.whl && \
    pip install https://publish.djl.ai/tritonserver/r23.09/tritontoolkit-23.9-py310-none-any.whl && \
    mkdir -p /opt/tritonserver/lib && mkdir -p /opt/tritonserver/backends/tensorrtllm && \
    curl -o /opt/tritonserver/lib/libtritonserver.so https://publish.djl.ai/tritonserver/r23.09/libtritonserver.so && \
    curl -o /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so https://publish.djl.ai/tensorrt-llm/0.5.0/libtriton_tensorrtllm.so && \
    cp /usr/local/lib/python${python_version}/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so /opt/tritonserver/lib/libnvinfer_plugin_tensorrt_llm.so.9 && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Final steps
RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_s5cmd.sh x64 && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} tensorrtllm" > /opt/djl/bin/telemetry && \
    scripts/patch_oss_dlc.sh python && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.tensorrtllm="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-25-0.tensorrtllm="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
