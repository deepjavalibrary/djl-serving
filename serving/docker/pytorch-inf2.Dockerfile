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
FROM ubuntu:20.04
ARG djl_version=0.26.0~SNAPSHOT
ARG torch_version=1.13.1
ARG python_version=3.9
ARG neuronsdk_version=2.16.0
ARG torch_neuronx_version=1.13.1.1.12.1
ARG transformers_neuronx_version=0.9.474
ARG neuronx_distributed_version=0.6.0
ARG neuronx_cc_version=2.12.54.0
ARG protobuf_version=3.19.6
ARG transformers_version=4.35.0
ARG accelerate_version=0.23.0
ARG diffusers_version=0.22.0
ARG pydantic_version=1.10.13
ARG optimum_neuron_version=0.0.16
EXPOSE 8080

# Sets up Path for Neuron tools
ENV PATH="/opt/aws/neuron/bin:${PATH}"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# ENV NO_OMP_NUM_THREADS=true
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV NEURON_SDK_PATH=/usr/local/lib/python3.9/dist-packages/torch_neuronx/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEURON_SDK_PATH
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.9/dist-packages/torch/lib
ENV PYTORCH_EXTRA_LIBRARY_PATH=$NEURON_SDK_PATH/libtorchneuron.so
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=1.13.1
ENV JAVA_OPTS="-Xmx1g -Xms1g -Xss2m -XX:+ExitOnOutOfMemoryError"
ENV NEURON_CC_FLAGS="--logfile /tmp/compile.log --temp-dir=/tmp"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/djl/partition && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/
COPY partition /opt/djl/partition
RUN mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} inf2" > /opt/djl/bin/telemetry && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_djl_serving.sh $djl_version && \
    scripts/install_inferentia2.sh && \
    pip install transformers==${transformers_version} accelerate==${accelerate_version} safetensors \
    neuronx-cc==${neuronx_cc_version} torch-neuronx==${torch_neuronx_version} transformers-neuronx==${transformers_neuronx_version} \
    neuronx_distributed==${neuronx_distributed_version} protobuf==${protobuf_version} sentencepiece jinja2 \
    diffusers==${diffusers_version} opencv-contrib-python-headless  Pillow --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    pydantic==${pydantic_version} optimum optimum-neuron==${optimum_neuron_version} && \
    scripts/install_s5cmd.sh x64 && \
    scripts/patch_oss_dlc.sh python && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.inf2="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-26-0.inf2="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL neuronsdk-version=$neuronsdk_version
