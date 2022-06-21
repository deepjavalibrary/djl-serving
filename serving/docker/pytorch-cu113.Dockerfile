ARG version=11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.18.0~SNAPSHOT
ARG torch_version=1.11.0

RUN mkdir -p /opt/djl/conf
COPY scripts scripts/
COPY config.properties /opt/djl/conf/
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    scripts/install_djl_serving.sh $djl_version && \
    scripts/install_python.sh && \
    pip3 install numpy && pip3 install torch==${torch_version} --extra-index-url https://download.pytorch.org/whl/cu113 && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV MODEL_SERVER_HOME=/opt/djl
ENV PYTORCH_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib
ENV PYTORCH_PRECXX11=true
ENV PYTORCH_VERSION=${torch_version}
ENV JAVA_OPTS="-Dai.djl.pytorch.num_interop_threads=1 -Dai.djl.default_engine=PyTorch"

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"
