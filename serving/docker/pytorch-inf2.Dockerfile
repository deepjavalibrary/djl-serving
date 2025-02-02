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
FROM ubuntu:22.04
ARG djl_version
ARG djl_serving_version
ARG python_version=3.10

# PyTorch and Vision
ARG torch_version=2.1.2
ARG torchvision_version=0.16.2

# Neuron SDK components
ARG neuronsdk_version=2.21.0
ARG torch_neuronx_version=2.1.2.2.4.0
ARG transformers_neuronx_version=0.13.322
ARG neuronx_distributed_version=0.10.0
ARG neuronx_distributed_inference_version=0.1.0
ARG neuronx_cc_version=2.16.345.0
ARG neuronx_cc_stubs_version=2.16.345.0
ARG torch_xla_version=2.1.6
ARG libneuronxla_version=2.1.681.0

ARG transformers_version=4.45.2
ARG tokenizers_version=0.20.3
ARG accelerate_version=0.29.2
ARG diffusers_version=0.28.2
ARG pydantic_version=2.6.1
ARG optimum_neuron_version=0.0.27
ARG huggingface_hub_version=0.25.2
# %2B is the url escape for the '+' character
ARG vllm_wheel="https://publish.djl.ai/neuron_vllm/vllm_neuronx-0.6.2%2Bneuron216-py3-none-any.whl"
EXPOSE 8080

# Sets up Path for Neuron tools
ENV PATH="/opt/aws/neuron/bin:${PATH}"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
# ENV NO_OMP_NUM_THREADS=true
ENV MODEL_SERVER_HOME=/opt/djl
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV NEURON_SDK_PATH=/usr/local/lib/python3.10/dist-packages/torch_neuronx/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEURON_SDK_PATH
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
ENV PYTORCH_EXTRA_LIBRARY_PATH=$NEURON_SDK_PATH/libtorchneuron.so
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=2.1.2
ENV JAVA_OPTS="-Xmx1g -Xms1g -Xss2m -XX:+ExitOnOutOfMemoryError"
ENV NEURON_CC_FLAGS="--logfile /tmp/compile.log --temp-dir=/tmp"
ENV SERVING_FEATURES=vllm,lmi-dist,tnx
# Making s5cmd discoverable
ENV PATH="/opt/djl/bin:${PATH}"

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
    echo "${djl_serving_version} inf2" > /opt/djl/bin/telemetry && \
    # Install python and djl serving
    scripts/install_python.sh && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version ${torch_version} && \
    # Install inferentia packages
    scripts/install_inferentia2.sh && \
    # Configure pip and install python packages
    pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com" && \
    pip install \
    accelerate==${accelerate_version} \
    safetensors \
    torchvision==${torchvision_version} \
    neuronx-cc==${neuronx_cc_version} \
    torch-neuronx==${torch_neuronx_version} \
    torch_xla==${torch_xla_version} \
    neuronx-cc-stubs==${neuronx_cc_stubs_version} \
    huggingface-hub==${huggingface_hub_version} \
    libneuronxla==${libneuronxla_version} \
    neuronx_distributed==${neuronx_distributed_version} \
    protobuf \
    sentencepiece \
    jinja2 \
    diffusers==${diffusers_version} \
    opencv-contrib-python-headless \
    Pillow \
    pydantic==${pydantic_version}  \
    optimum  \
    tiktoken  \
    blobfile && \
    # Install packages with no-deps flag
    pip install --no-deps \
    neuronx_distributed_inference==${neuronx_distributed_inference_version} \
    optimum-neuron==${optimum_neuron_version} \
    transformers-neuronx==${transformers_neuronx_version} \
    transformers==${transformers_version} \
    tokenizers==${tokenizers_version} && \
    # Install vllm and uninstall triton \
    pip install ${vllm_wheel} &&\
    echo y | pip uninstall triton && \
    # Install s5cmd and patch OSS DLC
    scripts/install_s5cmd.sh x64 && \
    scripts/patch_oss_dlc.sh python && \
    # Create user and set permissions
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    # Cleanup
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.inf2="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.inf2="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL neuronsdk-version=$neuronsdk_version
