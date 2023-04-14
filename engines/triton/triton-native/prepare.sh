#!/usr/bin/env bash

if [[ ! -d "tritonserver" ]]; then
  mkdir -p tritonserver/include
  cd tritonserver/include
  curl -O https://raw.githubusercontent.com/triton-inference-server/core/main/include/triton/core/tritonserver.h
  cd ../../
fi

rm -rf build
mkdir -p build/classes && cd build

javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/triton/jni/TritonLibrary.java -h include -d classes
cd ..