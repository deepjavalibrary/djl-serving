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

if [[ "$model_path" == "no_code" ]]; then
  unset model_path
fi

get_instance_type() {
  local token=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
  local instance_type=$(curl -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.instanceType')
  echo "$instance_type"
}

is_p4d_or_p5() {
  local instance_type=$(get_instance_type)
  if [[ "$instance_type" == *"p4d"* || "$instance_type" == *"p5"* ]]; then
    echo "true"
  else
    echo "false"
  fi
}

support_nvme() {
    local instance_type=$(get_instance_type)
    if [[ "$instance_type" == *"p4d"* || "$instance_type" == *"p5"* || "$instance_type" == *"g5"* || "$instance_type" == *"g6"* ]]; then
      echo "true"
    else
      echo "false"
    fi
}

if [[ "$(support_nvme)" == *"true"* ]]; then
  sudo rm -rf /opt/dlami/nvme/inf_tmp || true
  sudo mkdir -p /opt/dlami/nvme/inf_tmp && sudo chmod 777 /opt/dlami/nvme/inf_tmp
  nvme="/opt/dlami/nvme/inf_tmp:/tmp"
fi

is_llm=false
if [[ "$platform" == *"-gpu"* ]]; then # if the platform has cuda capabilities
  runtime="nvidia"
elif [[ "$platform" == *"lmi"* || "$platform" == *"trtllm"* || "$platform" == *"tensorrt-llm"* ]]; then # Runs multi-gpu
  runtime="nvidia"
  is_llm=true
  if [[ "$(is_p4d_or_p5)" == *"true"* ]]; then
    shm="20gb"
  else
    shm="12gb"
  fi
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
  for ((i = 0; i < $devices; i++)); do
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
    ${model_path:+-v ${model_path}/test:/opt/ml/input/data/training} \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/root/.aws \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    ${nvme:+-v ${nvme}} \
    ${env_file} \
    -e TEST_TELEMETRY_COLLECTION='true' \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" \
    ${args}

  exit 0
elif [[ "$docker_image" == *"text-generation-inference"* ]]; then
  container_id=$(docker run \
    -itd \
    --rm \
    -p 8080:80 \
    ${model_path:+-v ${model_path}:/opt/ml/model:ro} \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    ${nvme:+-v ${nvme}} \
    ${env_file} \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    "${docker_image}" \
    ${args})
else
  echo "$(whoami), UID: $UID"
  if [[ "$UID" == "1000" ]]; then
    uid_mapping="-u djl"
  fi
  container_id=$(docker run \
    -itd \
    --rm \
    --network="host" \
    ${model_path:+-v ${model_path}:/opt/ml/model:ro} \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/home/djl/.aws \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    ${nvme:+-v ${nvme}} \
    ${env_file} \
    -e TEST_TELEMETRY_COLLECTION='true' \
    -e SERVING_OPTS='-Dai.djl.logging.level=debug' \
    $uid_mapping \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" \
    ${args})
fi

set +x

echo "Launching ${container_id}..."

total_retries=24
if $is_llm; then
  total_retries=60
  if [[ "$platform" == *"inf2"* ]]; then
    total_retries=160
  fi
  if [[ "$platform" == *"trtllm"* || "$platform" == *"tensorrt-llm"* ]]; then
    total_retries=150
    echo "extra sleep of 10 min for trtllm compilation"
  fi
  if [[ "$platform" == *"trtllm-sq"* ]]; then
    echo "extra sleep of 15 min for smoothquant calibration"
    total_retries=140
  fi
  echo "extra sleep for 2 min on LLM models"
  sleep 120
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
  if [[ "$retry" -ge "$total_retries" ]]; then
    echo "Max retry exceeded."
    exit 1
  fi

  sleep 15
  ((++retry))
done
