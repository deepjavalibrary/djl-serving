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
ARG version=12.1.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:$version
ARG cuda_version=cu121
ARG djl_version=0.27.0~SNAPSHOT
# Base Deps
ARG python_version=3.10
ARG torch_version=2.1.2
ARG torch_vision_version=0.16.2
ARG pydantic_version=2.6.1
# HF Deps
ARG protobuf_version=3.20.3
ARG transformers_version=4.38.1
ARG accelerate_version=0.27.2
ARG diffusers_version=0.16.0
ARG bitsandbytes_version=0.41.1
ARG optimum_version=1.15.0
ARG auto_gptq_version=0.5.1
ARG datasets_version=2.17.1
# DeepSpeed Deps
ARG deepspeed_version=nightly
ARG deepspeed_wheel="https://publish.djl.ai/deepspeed/deepspeed-${deepspeed_version}-cp310-cp310-linux_x86_64.whl"
# LMI-Dist Deps
ARG vllm_wheel="https://publish.djl.ai/vllm/cu121-pt212/vllm-0.3.2-cp310-cp310-linux_x86_64.whl"
ARG flash_attn_wheel="https://publish.djl.ai/flash_attn/flash_attn_1-1.0.9-cp310-cp310-linux_x86_64.whl"
ARG dropout_layer_norm_wheel="https://publish.djl.ai/flash_attn/dropout_layer_norm-0.1-cp310-cp310-linux_x86_64.whl"
ARG rotary_emb_wheel="https://publish.djl.ai/flash_attn/rotary_emb-0.1-cp310-cp310-linux_x86_64.whl"
ARG flash_attn_2_wheel="https://publish.djl.ai/flash_attn/flash_attn-2.3.0-cp310-cp310-linux_x86_64.whl"
ARG lmi_vllm_wheel="https://publish.djl.ai/lmi_vllm/lmi_vllm-0.1.1-cp310-cp310-linux_x86_64.whl"
ARG lmi_dist_wheel="https://publish.djl.ai/lmi_dist/lmi_dist-nightly-py3-none-any.whl"
ARG awq_wheel="https://publish.djl.ai/awq/awq_inference_engine-0.0.0-cp310-cp310-linux_x86_64.whl"
ARG seq_scheduler_wheel="https://publish.djl.ai/seq_scheduler/seq_scheduler-0.1.0-py3-none-any.whl"
ARG peft_wheel="https://publish.djl.ai/peft/peft-0.5.0alpha-py3-none-any.whl"
ARG mmaploader_wheel="https://publish.djl.ai/mmaploader/mmaploader-nightly-py3-none-any.whl"
ARG aiccl_wheel="https://publish.djl.ai/aiccl/aiccl-1.1%2Bcu121torch2.1-cp310-cp310-linux_x86_64.whl"

EXPOSE 8080

COPY dockerd-entrypoint-with-cuda-compat.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# ENV NO_OMP_NUM_THREADS=true
ENV JAVA_OPTS="-Xmx1g -Xms1g -XX:+ExitOnOutOfMemoryError -Dai.djl.util.cuda.fork=true"
ENV MODEL_SERVER_HOME=/opt/djl
ENV MODEL_LOADING_TIMEOUT=1200
ENV PREDICT_TIMEOUT=240
ENV DJL_CACHE_DIR=/tmp/.djl.ai
ENV HF_HOME=/tmp/.cache/huggingface
ENV PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache
ENV BITSANDBYTES_NOWELCOME=1
ENV USE_AICCL_BACKEND=true
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV SAFETENSORS_FAST_GPU=1
ENV NCCL_BLOCKING_WAIT=0
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV SERVING_FEATURES=vllm,lmi-dist

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

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev g++ && \
    scripts/install_djl_serving.sh $djl_version && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} deepspeed" > /opt/djl/bin/telemetry && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_s5cmd.sh x64 && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==${torch_version} torchvision==${torch_vision_version} --extra-index-url https://download.pytorch.org/whl/cu121 \
    ${deepspeed_wheel} ${seq_scheduler_wheel} ${peft_wheel} ${mmaploader_wheel} ${aiccl_wheel} protobuf==${protobuf_version} \
    transformers==${transformers_version} hf-transfer zstandard datasets==${datasets_version} \
    mpi4py sentencepiece tiktoken einops accelerate==${accelerate_version} bitsandbytes==${bitsandbytes_version} \
    optimum==${optimum_version} auto-gptq==${auto_gptq_version} pandas pyarrow jinja2 \
    diffusers[torch]==${diffusers_version} opencv-contrib-python-headless safetensors scipy && \
    pip3 cache purge

RUN pip3 install ${flash_attn_wheel} ${dropout_layer_norm_wheel} ${rotary_emb_wheel} && \
    pip3 install ${flash_attn_2_wheel} ${lmi_dist_wheel} ${awq_wheel} ${lmi_vllm_wheel} ${vllm_wheel} pydantic==${pydantic_version} && \
    pip3 cache purge

# Add CUDA-Compat
RUN apt-get update && apt-get install -y cuda-compat-12-1 && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh deepspeed && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.deepspeed="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-26-0.deepspeed="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL deepspeed-version=$deepspeed_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver, CUDA 12.1 can work on this one too
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
