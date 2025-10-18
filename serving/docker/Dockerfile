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
ARG djl_version
ARG djl_serving_version

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf && \
    mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

RUN scripts/install_djl_serving.sh $djl_version $djl_serving_version && \
    scripts/install_s5cmd.sh x64 && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_serving_version} cpu" > /opt/djl/bin/telemetry && \
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
# Making s5cmd discoverable
ENV PATH="/opt/djl/bin:${PATH}"

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
LABEL djl-serving-version=$djl_serving_version


FROM base AS cpu-full

ARG python_version=3.11
ARG torch_version=2.7.1
ARG onnx_version=1.20.0
ARG sklearn_version=1.7.2
ARG xgboost_version=3.0.5
ARG pydantic_version=2.12.2

ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python${python_version}/dist-packages/torch/lib
ENV PYTORCH_VERSION=${torch_version}
ENV PYTORCH_FLAVOR=cpu

COPY scripts scripts/
RUN scripts/install_python.sh ${python_version} && \
    scripts/install_djl_serving.sh $djl_version $djl_serving_version $torch_version && \
    djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:$djl_version && \
    djl-serving -i com.microsoft.onnxruntime:onnxruntime:$onnx_version && \
    djl-serving -i ai.djl:basicdataset:$djl_version && \
    scripts/patch_oss_dlc.sh python && \
    pip3 install torch=="${torch_version}" torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install scikit-learn=="${sklearn_version}" skops cloudpickle xgboost=="${xgboost_version}" pydantic=="${pydantic_version}" && \
    echo "${djl_serving_version} cpufull" > /opt/djl/bin/telemetry && \
    rm -rf /opt/djl/logs && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.cpu-full="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-34-0.cpu-full="true"
LABEL torch-version=$torch_version
