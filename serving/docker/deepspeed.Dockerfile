ARG version=11.3.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.17.0
ARG torch_version=1.11.0
ARG deepspeed_version=0.6.5
ARG transformers_version=4.19.2

EXPOSE 8080

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

COPY scripts scripts/
RUN mkdir -p /opt/djl/conf
RUN mkdir -p /opt/djl/deps
COPY config.properties /opt/djl/conf/

RUN scripts/install_djl_serving.sh $djl_version
RUN scripts/install_python.sh

### Deep Speed installations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libaio-dev libopenmpi-dev && \
    pip3 install torch==${torch_version} --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip3 install deepspeed==${deepspeed_version} transformers==${transformers_version} triton==1.0.0 mpi4py

RUN rm -rf scripts
RUN apt-get clean -y && rm -rf /var/lib/apt/lists/*

LABEL maintainer="djl-dev@amazon.com"
