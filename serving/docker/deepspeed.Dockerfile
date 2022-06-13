ARG version=11.3.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:$version
ARG djl_version=0.17.0
ARG torch_version=1.11.0
ARG deepspeed_version=0.6.5
ARG transformers_version=4.19.2

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    openjdk-11-jdk-headless \
    curl

### Deep Speed installations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip libaio-dev libopenmpi-dev && \
    pip3 install torch==${torch_version} --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip3 install deepspeed==${deepspeed_version} transformers==${transformers_version} triton==1.0.0 mpi4py

RUN curl -O https://publish.djl.ai/djl-serving/djl-serving_${djl_version}-1_all.deb && \
    dpkg -i djl-serving_${djl_version}-1_all.deb && \
    rm djl-serving_${djl_version}-1_all.deb

RUN mkdir -p /opt/djl
COPY config.properties /opt/djl/conf/
RUN cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="lanking520@live.com"
