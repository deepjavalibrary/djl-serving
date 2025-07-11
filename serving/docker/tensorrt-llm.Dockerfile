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
ARG version=12.8.1-devel-ubuntu24.04
FROM nvidia/cuda:$version
ARG cuda_version=cu128
ARG python_version=3.12
ARG trtllm_version=0.20.0rc0
ARG djl_version
ARG djl_serving_version

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
ENV HF_HOME=/tmp/.cache/huggingface
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV BITSANDBYTES_NOWELCOME=1
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:/usr/local/lib/python${python_version}/dist-packages/tensorrt_libs:/usr/local/lib/python${python_version}/dist-packages/tensorrt_llm/libs/:${LD_LIBRARY_PATH}
ENV SERVING_FEATURES=trtllm
# Making s5cmd discoverable
ENV PATH="/opt/djl/bin:${PATH}"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN chmod -R +x scripts
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/djl/deps && \
    mkdir -p /opt/djl/partition && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition

# Install OpenMPI and other deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y g++ wget unzip openmpi-bin libopenmpi-dev libffi-dev git-lfs rapidjson-dev graphviz cuda-compat-12-8 && \
    scripts/install_python.sh ${python_version} && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# We install TRTLLM separately because it is hosted in a different pypi index
RUN pip install tensorrt_llm==${trtllm_version} --extra-index-url https://pypi.nvidia.com && \
    pip install tensorrt_llm==${trtllm_version} uvloop ninja

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

# Final steps
RUN scripts/install_djl_serving.sh $djl_version $djl_serving_version && \
    scripts/install_s5cmd.sh x64 && \
    scripts/security_patch.sh trtllm && \
    scripts/patch_oss_dlc.sh python && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_serving_version} tensorrtllm" > /opt/djl/bin/telemetry && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.tensorrtllm="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-34-0.tensorrtllm="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL trtllm-version=$trtllm_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
