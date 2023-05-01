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

is_partition=false
if [[ $4 == "partition" ]] || [[ $4 == "train" ]]; then
  is_partition=true
fi

is_llm=false
if [[ "$platform" == *"cu1"* ]]; then # if the platform has cuda capabilities
  runtime="nvidia"
elif [[ "$platform" == *"deepspeed"* ]]; then # Runs multi-gpu
  runtime="nvidia"
  is_llm=true
  shm="2gb"
elif [[ "$platform" == *"inf1"* ]]; then # if the platform is inferentia
  host_device="--device /dev/neuron0"
elif [[ "$platform" == *"inf2"* ]]; then # inf2: pytorch-inf2-24 24 will be the total devices
  OIFS=$IFS
  IFS='-'
  read -a strarr <<<"$platform"
  devices=${strarr[2]}
  IFS=$OIFS
  if [[ $devices -gt 1 ]]; then
    is_llm=true
  fi
  for ((i=0; i<$devices; i++))
  do
    host_device+=" --device /dev/neuron${i}"
  done
fi

if [[ -f ${PWD}/docker_env ]]; then
  env_file="--env-file ${PWD}/docker_env"
fi

rm -rf logs
mkdir logs

set -x
# start the docker container

if $is_partition; then
  docker run \
    -t \
    --rm \
    --network="host" \
    -v ${model_path}/test:/opt/ml/input/data/training \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/root/.aws \
    ${env_file} \
    -e TEST_TELEMETRY_COLLECTION='true' \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" \
    ${args}

  exit 0
else
  echo "$(whoami), UID: $UID"
  if [[ "$UID" == "1000" ]]; then
    uid_mapping="-u djl"
  fi
  container_id=$(docker run \
    -itd \
    --rm \
    --network="host" \
    -v ${model_path}:/opt/ml/model \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/home/djl/.aws \
    ${env_file} \
    -e TEST_TELEMETRY_COLLECTION='true' \
    $uid_mapping \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" \
    ${args})
fi

set +x

echo "Launching ${container_id}..."

total=24
if $is_llm; then
  echo "extra sleep for 5 min on LLM models"
  total=36
  sleep 300
fi

# retrying to connect, till djl serving started.
retry=0
while true; do
  echo "Start pinging to the host... Retry: $retry"
  http_code=$(curl -s -w '%{http_code}' -m 3 -o /dev/null "http://127.0.0.1:8080/ping" || true)
  if [[ "$http_code" -eq 200 ]]; then
    echo "DJL serving started"
    break
  fi
  if [[ "$(docker ps | wc -l)" == "1" ]]; then
    echo "Docker container shut down"
    exit 1
  fi
  if [[ "$retry" -ge "$total" ]]; then
    echo "Max retry exceeded."
    exit 1
  fi

  sleep 15
  ((++retry))
done
