FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
ARG ompi_version=4.1.4
ARG PYTHON_VERSION=3.10

# Install OpenMPI
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget && apt-get install -y ssh\
    && mkdir ompi && cd ompi \
    && wget -q -O - https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${ompi_version}.tar.gz | tar xzf - \
    && cd openmpi-${ompi_version} \
    && ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/ --with-cuda \
    && make -j"$(nproc)" install \
    && cd ../../ && rm -rf ompi \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends curl software-properties-common git && add-apt-repository -y ppa:deadsnakes/ppa && apt-get autoremove -y python3 && apt-get install -y "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-distutils" "python${PYTHON_VERSION}-venv" && ln -sf /usr/bin/"python${PYTHON_VERSION}" /usr/bin/python3 && ln -sf /usr/bin/"python${PYTHON_VERSION}" /usr/bin/python && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -rf get-pip.py

COPY requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt
RUN pip uninstall -y tensorrt

# Download and install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -P /workspace
RUN tar -xvf /workspace/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -C /usr/local/ && mv /usr/local/TensorRT-9.0.1.4 /usr/local/tensorrt
RUN pip install /usr/local/tensorrt/python/tensorrt-9.0.1*cp310-none-linux_x86_64.whl && rm -fr /workspace/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:$LD_LIBRARY_PATH

# Download and install polygraphy, only required if you need to run TRT-LLM python tests
RUN pip install https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/polygraphy-0.48.1-py2.py3-none-any.whl

COPY ./build/tensorrt_llm*.whl /tmp/
RUN pip install /tmp/tensorrt_llm*.whl 
