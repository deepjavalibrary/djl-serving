FROM ubuntu:20.04
ARG djl_version=0.17.0

RUN mkdir -p /opt/djl/conf
COPY scripts scripts/
COPY config.properties /opt/djl/conf/
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    scripts/install_djl_serving.sh $djl_version && \
    rm -rf scripts && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/djl

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV MXNET_ENGINE_TYPE=NaiveEngine
ENV JAVA_OPTS="-Dai.djl.pytorch.num_interop_threads=1"
ENV MODEL_SERVER_HOME=/opt/djl

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="djl-dev@amazon.com"
