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
ARG version=12.2.2-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:$version
ARG cuda_version=cu122
ARG python_version=3.10
ARG TORCH_VERSION=2.1.0
ARG djl_version=0.24.0~SNAPSHOT
ARG transformers_version=4.34.0
ARG accelerate_version=0.23.0
ARG tensorrtlibs_version=9.1.0.post12.dev4
ARG trtllm_toolkit_version=nightly
ARG cuda_python_version=12.2.0
ARG peft_wheel="https://publish.djl.ai/peft/peft-0.5.0alpha-py3-none-any.whl"
ARG trtllm_toolkit_wheel="https://publish.djl.ai/tensorrt-llm/toolkit/tensorrt_llm_toolkit-${trtllm_toolkit_version}-py3-none-any.whl"
ARG trtllm_wheel="https://djl-ai.s3.amazonaws.com/publish/tensorrt-llm/0.5.0/tensorrt_llm-0.5.0-py3-none-any.whl"
ARG triton_toolkit_wheel="https://publish.djl.ai/tritonserver/r23.09/tritontoolkit-23.9-py310-none-any.whl"
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
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:/usr/local/lib/python${python_version}/dist-packages/tensorrt_libs:${LD_LIBRARY_PATH}
ENV LMI_BACKEND=TRT-LLM

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

# Install PyTorch and TensorRT
RUN pip install torch==${TORCH_VERSION} transformers==${transformers_version} accelerate==${accelerate_version} ${peft_wheel} sentencepiece \
    mpi4py cuda-python==${cuda_python_version} onnx polygraphy && \
    pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt==${tensorrtlibs_version} && \
    pip install --no-deps ${trtllm_wheel} && \
    pip3 cache purge

# download dependencies
RUN pip install ${triton_toolkit_wheel} ${trtllm_toolkit_wheel} && \
    mkdir -p /opt/tritonserver/lib && mkdir -p /opt/tritonserver/backends/tensorrtllm && \
    curl -o /opt/tritonserver/lib/libtritonserver.so https://publish.djl.ai/tritonserver/r23.09/libtritonserver.so && \
    curl -o /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so https://publish.djl.ai/tensorrt-llm/0.5.0/libtriton_tensorrtllm.so && \
    curl -o /opt/tritonserver/lib/libnvinfer_plugin_tensorrt_llm.so.9 https://publish.djl.ai/tensorrt-llm/0.5.0/libnvinfer_plugin_tensorrt_llm.so.9 && \
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
LABEL djl-version=$djl_version
LABEL trtllm-version=$trtllm_version
LABEL cuda-version=$cuda_version
