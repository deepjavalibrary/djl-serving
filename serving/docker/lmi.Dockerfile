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
ARG djl_version=0.28.0
# Base Deps
ARG python_version=3.10
ARG torch_version=2.3.0
ARG torch_vision_version=0.18.0
ARG onnx_version=1.18.0
ARG onnxruntime_wheel="https://publish.djl.ai/onnxruntime/1.18.0/onnxruntime_gpu-1.18.0-cp310-cp310-linux_x86_64.whl"
ARG pydantic_version=2.7.1
ARG djl_converter_wheel="https://publish.djl.ai/djl_converter/djl_converter-0.28.0-py3-none-any.whl"
ARG vllm_cuda_name="cu12"
ARG vllm_nccl_version=2.18.1
# HF Deps
ARG protobuf_version=3.20.3
ARG transformers_version=4.41.1
ARG accelerate_version=0.30.1
ARG bitsandbytes_version=0.43.1
ARG optimum_version=1.20.0
ARG auto_gptq_version=0.7.1
ARG datasets_version=2.19.1
ARG autoawq_version=0.2.5
# LMI-Dist Deps
ARG vllm_wheel="https://publish.djl.ai/vllm/cu124-pt230/vllm-0.4.2%2Bcu124-cp310-cp310-linux_x86_64.whl"
ARG flash_attn_2_wheel="https://publish.djl.ai/flash_attn/cu124-pt230/flash_attn-2.5.8-cp310-cp310-linux_x86_64.whl"
ARG lmi_dist_wheel="https://publish.djl.ai/lmi_dist/lmi_dist-10.0.0-py3-none-any.whl"
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
ENV VLLM_CONFIG_ROOT=/opt/djl/vllm/.config


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
    rm -f /usr/local/djl-serving-*/lib/onnxruntime-1.*.jar && \
    curl -o $(ls -d /usr/local/djl-serving-*/)lib/onnxruntime_gpu-$onnx_version.jar https://publish.djl.ai/onnxruntime/$onnx_version/onnxruntime_gpu-$onnx_version.jar && \
    mkdir -p /opt/djl/bin && cp scripts/telemetry.sh /opt/djl/bin && \
    echo "${djl_version} lmi" > /opt/djl/bin/telemetry && \
    scripts/install_djl_serving.sh $djl_version ${torch_version} && \
    scripts/install_python.sh ${python_version} && \
    scripts/install_s5cmd.sh x64 && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==${torch_version} torchvision==${torch_vision_version} --extra-index-url https://download.pytorch.org/whl/cu121 \
    ${seq_scheduler_wheel} peft==${peft_version} protobuf==${protobuf_version} \
    transformers==${transformers_version} hf-transfer zstandard datasets==${datasets_version} \
    mpi4py sentencepiece tiktoken blobfile einops accelerate==${accelerate_version} bitsandbytes==${bitsandbytes_version} \
    optimum==${optimum_version} auto-gptq==${auto_gptq_version} pandas pyarrow jinja2 \
    opencv-contrib-python-headless safetensors scipy onnx sentence_transformers ${onnxruntime_wheel} autoawq==${autoawq_version} && \
    pip3 install ${djl_converter_wheel} --no-deps && \
    pip3 cache purge

RUN pip3 install ${flash_attn_2_wheel} ${lmi_dist_wheel} ${vllm_wheel} pydantic==${pydantic_version} && \
    pip3 cache purge

# Add CUDA-Compat
RUN apt-get update && apt-get install -y cuda-compat-12-4 && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# We use the same NCCL version as vLLM for lmi-dist https://github.com/vllm-project/vllm/blob/v0.4.2/vllm/utils.py#L641-L646
# This is due to https://github.com/vllm-project/vllm/blob/v0.4.2/vllm/distributed/device_communicators/pynccl.py#L1-L9
RUN mkdir -p ${VLLM_CONFIG_ROOT}/vllm/nccl/$vllm_cuda_name/ && curl -L -o ${VLLM_CONFIG_ROOT}/vllm/nccl/$vllm_cuda_name/libnccl.so.$vllm_nccl_version \
    https://github.com/vllm-project/vllm-nccl/releases/download/v0.1.0/$vllm_cuda_name-libnccl.so.$vllm_nccl_version && \
    # The following is done only so that we can run the CI with `-u djl`. Sagemaker wouldn't require this.
    chmod -R a+w ${VLLM_CONFIG_ROOT}

RUN scripts/patch_oss_dlc.sh python && \
    scripts/security_patch.sh lmi && \
    useradd -m -d /home/djl djl && \
    chown -R djl:djl /opt/djl && \
    rm -rf scripts && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.lmi="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-28-0.lmi="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver, CUDA 12.1 can work on this one too
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
