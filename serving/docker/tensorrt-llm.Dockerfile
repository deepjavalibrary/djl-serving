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
ARG version=12.5.1-devel-ubuntu22.04
FROM nvidia/cuda:$version
ARG cuda_version=cu125
ARG python_version=3.10
ARG TORCH_VERSION=2.4.0
ARG djl_version
ARG djl_serving_version
ARG transformers_version=4.44.2
ARG accelerate_version=0.32.1
ARG tensorrtlibs_version=10.1.0
# %2B is the url escape for the '+' character
ARG trtllm_toolkit_version=0.12.0%2Bnightly
ARG trtllm_version=v0.12.0
ARG cuda_python_version=12.5
ARG peft_version=0.10.0
ARG triton_version=r24.04
ARG trtllm_toolkit_wheel="https://publish.djl.ai/tensorrt-llm/toolkit/tensorrt_llm_toolkit-${trtllm_toolkit_version}-py3-none-any.whl"
ARG trtllm_wheel="https://publish.djl.ai/tensorrt-llm/${trtllm_version}/tensorrt_llm-0.12.0-cp310-cp310-linux_x86_64.whl"
ARG triton_toolkit_wheel="https://publish.djl.ai/tritonserver/${triton_version}/tritontoolkit-24.4-py310-none-any.whl"
ARG pydantic_version=2.6.1
ARG modelopt_version=0.15.0
ARG janus_version=1.0.0
ARG pynvml_verison=11.5.0
ARG numpy_version=1.26.4
ARG datasets_version=2.19.1

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
RUN apt-get update && apt-get install -y g++ wget unzip openmpi-bin libopenmpi-dev libffi-dev git-lfs rapidjson-dev graphviz && \
    scripts/install_python.sh ${python_version} && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install PyTorch
# Qwen needs transformers_stream_generator, tiktoken and einops
RUN pip install torch==${TORCH_VERSION} transformers==${transformers_version} accelerate==${accelerate_version} peft==${peft_version} sentencepiece \
    mpi4py cuda-python==${cuda_python_version} onnx polygraphy pynvml==${pynvml_verison} datasets==${datasets_version} pydantic==${pydantic_version} scipy torchprofile bitsandbytes ninja \
    transformers_stream_generator einops tiktoken jinja2 graphviz blobfile colored h5py strenum pulp flax easydict && \
    pip3 cache purge

# Install TensorRT and TRT-LLM Deps
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt==${tensorrtlibs_version} janus==${janus_version} nvidia-modelopt==${modelopt_version} && \
    pip install --no-deps ${trtllm_wheel} && \
    pyver=$(echo $python_version | awk -F. '{print $1$2}') && \
    pip3 cache purge

# download dependencies
RUN pip install ${triton_toolkit_wheel} ${trtllm_toolkit_wheel} && \
    mkdir -p /opt/tritonserver/lib && mkdir -p /opt/tritonserver/backends/tensorrtllm && \
    curl -o /opt/tritonserver/lib/libtritonserver.so https://publish.djl.ai/tritonserver/${triton_version}/libtritonserver.so && \
    curl -o /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so https://publish.djl.ai/tensorrt-llm/${trtllm_version}/libtriton_tensorrtllm.so && \
    curl -o /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm_common.so https://publish.djl.ai/tensorrt-llm/${trtllm_version}/libtriton_tensorrtllm_common.so && \
    curl -o /opt/tritonserver/lib/libnvinfer_plugin_tensorrt_llm.so.10 https://publish.djl.ai/tensorrt-llm/${trtllm_version}/libnvinfer_plugin_tensorrt_llm.so.10 && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

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
    pip3 install numpy==${numpy_version} && \
    pip3 cache purge && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add CUDA-Compat
RUN apt-get update && apt-get install -y cuda-compat-12-4 && apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
LABEL dlc_major_version="1"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.tensorrtllm="true"
LABEL com.amazonaws.ml.engines.sagemaker.dlc.framework.djl.v0-32-0.tensorrtllm="true"
LABEL com.amazonaws.sagemaker.capabilities.multi-models="true"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
LABEL djl-version=$djl_version
LABEL djl-serving-version=$djl_serving_version
LABEL trtllm-version=$trtllm_version
LABEL cuda-version=$cuda_version
# To use the 535 CUDA driver
LABEL com.amazonaws.sagemaker.inference.cuda.verified_versions=12.2
