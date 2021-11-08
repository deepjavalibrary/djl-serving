ARG version=0.14.0
FROM deepjavalibrary/djl-serving:$version

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        gnupg2 \
        wget \
        python3-pip \
        python3-setuptools \
    && cd /usr/local/bin \
    && pip3 --no-cache-dir install --upgrade pip \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN echo "deb https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Installing Neuron Tools
RUN apt-get update -y  \
    && apt-get install -y aws-neuron-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Include framework tensorflow-neuron or torch-neuron and compiler (compiler not needed for inference)
RUN pip3 install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install numpy \
    && pip3 install torch-neuron \
      --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    && wget -q https://publish.djl.ai/pytorch-1.9.1/jnilib/precxx11/0.14.0/linux-x86_64/cpu/libdjl_torch.so \
        -O /usr/local/lib/python3.6/dist-packages/torch/lib/libdjl_torch.so

# Sets up Path for Neuron tools
ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

ENV NEURON_SDK_PATH=/usr/local/lib/python3.6/dist-packages/torch_neuron/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEURON_SDK_PATH
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/torch/lib
ENV PYTORCH_EXTRA_LIBRARY_PATH=$NEURON_SDK_PATH/libtorchneuron.so
ENV PYTORCH_PRECXX11=true
ENV JAVA_OPTS="-Dai.djl.pytorch.num_interop_threads=1 -Dai.djl.default_engine=PyTorch"

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="frankfliu2000@gmail.com"
