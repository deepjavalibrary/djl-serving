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
ARG version=12.4.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:$version
ARG cuda_version=cu124
ARG djl_version=0.30.0~SNAPSHOT
# Base Deps
ARG python_version=3.10
ARG torch_version=2.3.1
ARG torch_vision_version=0.18.1
ARG onnx_version=1.18.0
ARG onnxruntime_wheel="https://publish.djl.ai/onnxruntime/1.18.0/onnxruntime_gpu-1.18.0-cp310-cp310-linux_x86_64.whl"
ARG pydantic_version=2.8.2
ARG djl_converter_wheel="https://publish.djl.ai/djl_converter/djl_converter-0.30.0-py3-none-any.whl"
# HF Deps
ARG protobuf_version=3.20.3
ARG transformers_version=4.43.2
ARG accelerate_version=0.32.1
ARG bitsandbytes_version=0.43.1
ARG optimum_version=1.21.2
ARG auto_gptq_version=0.7.1
ARG datasets_version=2.20.0
ARG autoawq_version=0.2.5
ARG tokenizers_version=0.19.1
# LMI-Dist Deps
ARG vllm_wheel="https://publish.djl.ai/vllm/cu124-pt231/vllm-0.5.3.post1%2Bcu124-cp310-cp310-linux_x86_64.whl"
ARG flash_attn_2_wheel="https://github.com/vllm-project/flash-attention/releases/download/v2.5.9.post1/vllm_flash_attn-2.5.9.post1-cp310-cp310-manylinux1_x86_64.whl"
ARG flash_infer_wheel="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.9/flashinfer-0.0.9+cu121torch2.3-cp310-cp310-linux_x86_64.whl"
# %2B is the url escape for the '+' character
ARG lmi_dist_wheel="https://publish.djl.ai/lmi_dist/lmi_dist-11.0.0%2Bnightly-py3-none-any.whl"
ARG seq_scheduler_wheel="https://publish.djl.ai/seq_scheduler/seq_scheduler-0.1.0-py3-none-any.whl"
ARG peft_version=0.11.1

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
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=${torch_version}
ENV PYTORCH_FLAVOR=cu121-precxx11
ENV VLLM_NO_USAGE_STATS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn


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
    && scripts/install_djl_serving.sh $djl_version \
    && scripts/install_djl_serving.sh $djl_version ${torch_version} \
    && djl-serving -i ai.djl.onnxruntime:onnxruntime-engine:$djl_version \
    && djl-serving -i com.microsoft.onnxruntime:onnxruntime_gpu:$onnx_version \
    && scripts/install_python.sh ${python_version} \
    && scripts/install_s5cmd.sh x64 \
    && mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin \
    && echo "${djl_version} lmi" > /opt/djl/bin/telemetry \
    && pip3 cache purge \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==${torch_version} torchvision==${torch_vision_version} --extra-index-url https://download.pytorch.org/whl/cu121 \
    ${seq_scheduler_wheel} peft==${peft_version} protobuf==${protobuf_version} \
    transformers==${transformers_version} hf-transfer zstandard datasets==${datasets_version} \
    mpi4py sentencepiece tiktoken blobfile einops accelerate==${accelerate_version} bitsandbytes==${bitsandbytes_version} \
    auto-gptq==${auto_gptq_version} pandas pyarrow jinja2 retrying \
    opencv-contrib-python-headless safetensors scipy onnx sentence_transformers ${onnxruntime_wheel} autoawq==${autoawq_version} \
    tokenizers==${tokenizers_version} pydantic==${pydantic_version} \
    # TODO: installing optimum here due to version conflict.
    && pip3 install ${djl_converter_wheel} optimum==${optimum_version} --no-deps \
    && git clone https://github.com/neuralmagic/AutoFP8.git && cd AutoFP8 && git reset --hard 4b2092c && pip3 install . && cd .. && rm -rf AutoFP8 \
    && pip3 cache purge

RUN pip3 install ${flash_attn_2_wheel} ${lmi_dist_wheel} ${vllm_wheel} ${flash_infer_wheel} \
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
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-30-0.lmi="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver, CUDA 12.1 can work on this one too
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
