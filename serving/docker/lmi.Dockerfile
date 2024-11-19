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
FROM nvidia/cuda:$version
ARG cuda_version=cu124
ARG djl_version
ARG djl_serving_version
# Base Deps
ARG python_version=3.11
ARG torch_version=2.5.1
ARG torch_vision_version=0.20.1
ARG djl_torch_version=2.5.1
ARG onnx_version=1.19.0
ARG pydantic_version=2.9.2
ARG djl_converter_wheel="https://publish.djl.ai/djl_converter/djl_converter-0.31.0-py3-none-any.whl"
# HF Deps
ARG protobuf_version=3.20.3
ARG transformers_version=4.45.2
ARG accelerate_version=1.0.1
ARG bitsandbytes_version=0.44.1
ARG optimum_version=1.23.2
ARG auto_gptq_version=0.7.1
ARG datasets_version=3.0.1
ARG autoawq_version=0.2.5
ARG tokenizers_version=0.20.1
# LMI-Dist Deps
ARG vllm_wheel="https://publish.djl.ai/vllm/cu124-pt251/vllm-0.6.3.post1%2Bcu124-cp311-cp311-linux_x86_64.whl"
ARG flash_infer_wheel="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu124torch2.4-cp311-cp311-linux_x86_64.whl"
# %2B is the url escape for the '+' character
ARG lmi_dist_wheel="https://publish.djl.ai/lmi_dist/lmi_dist-13.0.0%2Bnightly-py3-none-any.whl"
ARG seq_scheduler_wheel="https://publish.djl.ai/seq_scheduler/seq_scheduler-0.1.0-py3-none-any.whl"
ARG peft_version=0.13.2

ARG sagemaker_fast_model_loader_wheel="https://publish.djl.ai/fast-model-loader/sagemaker_fast_model_loader-0.1.0-cp311-cp311-linux_x86_64.whl"

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
ENV LD_LIBRARY_PATH=/usr/local/lib/python${python_version}/dist-packages/nvidia/cudnn/lib/
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
ENV SERVING_FEATURES=vllm,lmi-dist

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf \
    && mkdir -p /opt/djl/deps \
    && mkdir -p /opt/djl/partition \
    && mkdir -p /opt/ml/model
COPY config.properties /opt/djl/conf/config.properties
COPY partition /opt/djl/partition

COPY distribution[s]/ ./
RUN mv *.deb djl-serving_all.deb || true

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev g++ \
    && scripts/install_openssh.sh \
    && scripts/install_djl_serving.sh $djl_version $djl_serving_version \
    && scripts/install_djl_serving.sh $djl_version $djl_serving_version ${djl_torch_version} \
    && djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:$djl_version \
    && djl-serving -i com.microsoft.onnxruntime:onnxruntime_gpu:$onnx_version \
    && scripts/install_python.sh ${python_version} \
    && scripts/install_s5cmd.sh x64 \
    && mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin \
    && echo "${djl_serving_version} lmi" > /opt/djl/bin/telemetry \
    && pip3 cache purge \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==${torch_version} torchvision==${torch_vision_version} --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install \
    ${seq_scheduler_wheel} \
    peft==${peft_version} \
    protobuf==${protobuf_version} \
    transformers==${transformers_version} \
    hf-transfer \
    zstandard \
    datasets==${datasets_version} \
    mpi4py \
    sentencepiece \
    tiktoken \
    blobfile \
    einops \
    accelerate==${accelerate_version} \
    bitsandbytes==${bitsandbytes_version} \
    auto-gptq==${auto_gptq_version} \
    pandas \
    pyarrow \
    jinja2 \
    retrying \
    opencv-contrib-python-headless \
    safetensors \
    scipy \
    onnx \
    sentence_transformers \
    onnxruntime \
    autoawq==${autoawq_version} \
    tokenizers==${tokenizers_version} \
    pydantic==${pydantic_version} \
    ${djl_converter_wheel} \
    optimum==${optimum_version} \
    ${flash_infer_wheel} \
    ${vllm_wheel} \
    ${lmi_dist_wheel} \
    torch==${torch_version} \
    torchvision==${torch_vision_version} \
    ${sagemaker_fast_model_loader_wheel} \
    && git clone https://github.com/neuralmagic/AutoFP8.git && cd AutoFP8 && git reset --hard 4b2092c && pip3 install . && cd .. && rm -rf AutoFP8 \
    && pip3 cache purge

# Add CUDA-Compat
RUN apt-get update && apt-get install -y cuda-compat-12-4 && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN scripts/patch_oss_dlc.sh python \
    && scripts/security_patch.sh lmi \
    && useradd -m -d /home/djl djl \
    && chown -R djl:djl /opt/djl \
    && rm -rf scripts \
    && pip3 cache purge \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.lmi="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.lmi="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver, CUDA 12.4 can work on this one too
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
