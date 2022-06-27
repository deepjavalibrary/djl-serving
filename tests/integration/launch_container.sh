#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Bad number of arguments."
  echo "Usage: $0 <docker_image> <model_path> <platform> [args...]"
  exit 1
fi

docker_image=$1 #required
model_path=$2   #required
platform=$3     #required
args=${@:4}     #optional

if [[ "$platform" == *"cu113"* ]]; then # if the platform has cuda capabilities
  runtime="nvidia"
elif [[ "$platform" == *"inf1"* ]]; then # if the platform is inferentia
  host_device="/dev/neuron0"
fi

set -x
# start the docker container
docker run \
  -itd \
  --rm \
  -p 8080:8080 \
  -v ${model_path}:/opt/ml/model \
  -v ${PWD}/logs:/opt/djl/logs \
  ${runtime:+--runtime="${runtime}"} \
  ${host_device:+--device "${host_device}"} \
  "${docker_image}" \
  ${args}
set +x

# retrying to connect, till djl serving started.
retry=0
while true; do
  echo "Start pinging to the host... Retry: $retry"
  http_code=$(curl -s -w '%{http_code}' -o /dev/null "http://127.0.0.1:8080/ping" || true)
  if [[ "$http_code" -eq 200 ]]; then
    echo "DJL serving started"
    break
  fi
  if [[ "$retry" -ge 5 ]]; then
    echo "Max retry exceeded."
    exit 1
  fi

  sleep 15
  ((++retry))
done
