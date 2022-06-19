FROM ubuntu:20.04
ARG djl_version=0.17.0

COPY scripts scripts/
RUN scripts/install_djl_serving.sh $djl_version
RUN scripts/install_python.sh
RUN rm -rf scripts djl-serving_all.deb

RUN mkdir -p /opt/djl
COPY config.properties /opt/djl/conf/
RUN cp /usr/local/djl-serving-*/conf/log4j2.xml /opt/djl/conf/

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV MXNET_ENGINE_TYPE=NaiveEngine
ENV JAVA_OPTS="-Dai.djl.pytorch.num_interop_threads=1"
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="frankfliu2000@gmail.com"
