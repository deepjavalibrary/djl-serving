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
FROM ubuntu:22.04 AS base
ARG djl_version=0.28.0~SNAPSHOT

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

RUN scripts/install_djl_serving.sh $djl_version && \
    scripts/install_s5cmd.sh x64 && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} cpu" > /opt/djl/bin/telemetry && \
    scripts/security_patch.sh cpu && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    rm -rf scripts

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError"
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers

RUN useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.cpu="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version


FROM base AS cpu-full

ARG torch_version=2.2.2

COPY scripts scripts/
RUN scripts/install_python.sh && \
    scripts/install_djl_serving.sh $djl_version $torch_version && \
    echo "${djl_version} cpufull" > /opt/djl/bin/telemetry && \
    djl-serving -i ai.djl.pytorch:pytorch-native-cpu:$torch_version:linux-x86_64 && \
    djl-serving -i ai.djl.tensorflow:tensorflow-native-cpu:2.10.1:linux-x86_64 && \
    scripts/patch_oss_dlc.sh python && \
    rm -rf /opt/djl/logs && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.cpu-full="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-28-0.cpu-full="true"
LABEL torch-version=$torch_version


FROM ubuntu:22.04 as vllm-build
# Borrowed from https://github.com/vllm-project/vllm/blob/v0.4.3/Dockerfile.cpu
ARG vllm_version=v0.4.3

WORKDIR /usr/src

RUN apt-get update  -y \
    && apt-get install -y git wget vim numactl gcc-12 g++-12 python3 python3-pip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

RUN pip install --upgrade pip \
    && pip install wheel packaging ninja setuptools>=49.4.0 numpy

RUN git clone https://github.com/vllm-project/vllm -b ${vllm_version} && cd vllm && \
    pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu python3 setup.py bdist_wheel


FROM base AS lmi-cpu

ARG torch_version=2.3.0
ARG protobuf_version=3.20.3
ARG transformers_version=4.41.1
ARG accelerate_version=0.30.1
ARG datasets_version=2.19.1
ARG seq_scheduler_wheel="https://publish.djl.ai/seq_scheduler/seq_scheduler-0.1.0-py3-none-any.whl"
ARG peft_version=0.11.1

COPY scripts scripts/
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    scripts/install_python.sh && \
    scripts/install_djl_serving.sh $djl_version $torch_version && \
    scripts/install_s5cmd.sh x64 && \
    echo "${djl_version} cpufull" > /opt/djl/bin/telemetry && \
    djl-serving -i ai.djl.pytorch:pytorch-native-cpu:$torch_version:linux-x86_64 && \
    djl-serving -i ai.djl.tensorflow:tensorflow-native-cpu:2.10.1:linux-x86_64 && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY --from=vllm-build /usr/src/vllm/dist/*.whl scripts/

# FIXME remove Trtion import in the next release
RUN pip3 install torch==${torch_version}+cpu triton>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu \
    ${seq_scheduler_wheel} peft==${peft_version} protobuf==${protobuf_version} \
    transformers==${transformers_version} hf-transfer zstandard datasets==${datasets_version} \
    mpi4py sentencepiece tiktoken blobfile einops accelerate==${accelerate_version} scripts/vllm*.whl \
    jinja2 safetensors ninja scipy sentence_transformers && \
    pip3 cache purge

# final cleanup \
RUN scripts/patch_oss_dlc.sh python && \
    rm -rf /opt/djl/logs && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.lmi-cpu="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-28-0.lmi-cpu="true"
