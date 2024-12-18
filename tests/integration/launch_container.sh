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

echo launch_container.sh: using docker image: $docker_image

is_sm_neo_context=false
if [[ $4 == "sm_neo_context" ]]; then
  is_sm_neo_context=true
  if [[ $5 == "jumpstart_integration" ]]; then
    jumpstart_integration=true
  fi
fi

is_partition=false
if [[ $4 == "partition" ]] || [[ $4 == "train" ]]; then
  is_partition=true
fi

if [[ "$model_path" == "no_code" ]]; then
  unset model_path
fi

is_multi_node=false
if [[ $4 == "multi_node" ]]; then
  is_multi_node=true
fi

start_docker_network() {
  local subnet="$1"
  local name="$2"

  if [ $(docker network ls --format '{{.Name}}' | grep -q "$name") -eq 0 ]; then
    echo "Network '$name' already exists."
    return 0
  fi

  docker network create --subnet="$subnet" "$name"
  if [ $? -eq 0 ]; then
    echo "Network '$name' created successfully."
  else
    echo "Failed to create network '$name'."
    return 1
  fi
}

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
  if [[ "$(is_p4d_or_p5)" == *"true"* || $is_multi_node ]]; then
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

subnet="192.168.10.0/24"
network_name="docker-net"

leader_ip="192.168.10.2"
leader_hostname="lmi-0.lmi.default"
worker_ip="192.168.10.3"
worker_hostname="lmi-0-1.lmi.default"

get_aws_credentials() {
  TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
  ROLE_NAME=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -GET http://169.254.169.254/latest/meta-data/iam/security-credentials/)
  CREDENTIALS=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -GET http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)

  export AWS_ACCESS_KEY_ID=$(echo "$CREDENTIALS" | grep AccessKeyId | cut -d':' -f2 | tr -d ' ",' | sed 's/^"//' | sed 's/"$//')
  export AWS_SECRET_ACCESS_KEY=$(echo "$CREDENTIALS" | grep SecretAccessKey | cut -d':' -f2 | tr -d ' ",' | sed 's/^"//' | sed 's/"$//')
  export AWS_SESSION_TOKEN=$(echo "$CREDENTIALS" | grep Token | cut -d':' -f2 | tr -d ' ",' | sed 's/^"//' | sed 's/"$//')
}

# start the docker container
if $is_multi_node; then
  get_aws_credentials
  start_docker_network $subnet $network_name

  LWS_NAME=lmi
  GROUP_INDEX=0
  NAMESPACE=default
  docker run \
    -t \
    -d \
    --rm \
    --gpus '"device=0,1"' \
    -p 8080:8080 \
    --network=${network_name} \
    --ip=${leader_ip} \
    --hostname=${leader_hostname} \
    ${model_path:+-v ${model_path}:/opt/ml/model:ro} \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/root/.aws \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
    -e MASTER_ADDR=${leader_hostname} \
    -e LWS_NAME=${LWS_NAME} \
    -e GROUP_INDEX=${GROUP_INDEX} \
    -e NAMESPACE=${NAMESPACE} \
    -e DJL_CLUSTER_SIZE=2 \
    -e DJL_LEADER_ADDR=${leader_hostname} \
    -e DJL_WORKER_ADDR_FORMAT="${LWS_NAME}-${GROUP_INDEX}-%d.${LWS_NAME}.${NAMESPACE}" \
    ${env_file} \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" "service ssh start; djl-serving"

  docker run \
    -t \
    -d \
    --rm \
    --gpus '"device=2,3"' \
    --network=${network_name} \
    --ip=${worker_ip} \
    --hostname=${worker_hostname} \
    ${model_path:+-v ${model_path}:/opt/ml/model:ro} \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/root/.aws \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    -e DJL_LEADER_ADDR=${leader_hostname} \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
    ${env_file} \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}" "service ssh start; /usr/bin/python3 /opt/djl/partition/run_multi_node_setup.py 2>&1 | tee /opt/djl/logs/lmi-worker.log; tail -f"
elif $is_sm_neo_context; then
  docker run \
    -t \
    --rm \
    --network="host" \
    ${model_path:+-v ${model_path}/uncompiled:/opt/ml/model/input} \
    ${model_path:+-v ${model_path}/compiled:/opt/ml/model/compiled} \
    ${model_path:+-v ${model_path}/errors.json:/opt/ml/compilation/errors/errors.json} \
    ${model_path:+-v ${model_path}/cache:/opt/ml/compilation/cache} \
    -v ${PWD}/logs:/opt/djl/logs \
    -v ~/.aws:/root/.aws \
    -v ~/sagemaker_infra/:/opt/ml/.sagemaker_infra/:ro \
    ${jumpstart_integration:+-e SM_CACHE_JUMPSTART_FORMAT=true} \
    -e SM_NEO_EXECUTION_CONTEXT=1 \
    -e SM_NEO_INPUT_MODEL_DIR=/opt/ml/model/input \
    -e SM_NEO_COMPILED_MODEL_DIR=/opt/ml/model/compiled \
    -e SM_NEO_COMPILATION_ERROR_FILE=/opt/ml/compilation/errors/errors.json \
    -e SM_NEO_CACHE_DIR=/opt/ml/compilation/cache \
    -e SM_NEO_HF_CACHE_DIR=/opt/ml/compilation/cache \
    -e COMPILER_OPTIONS={} \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
    ${env_file} \
    ${runtime:+--runtime="${runtime}"} \
    ${shm:+--shm-size="${shm}"} \
    ${host_device:+ ${host_device}} \
    "${docker_image}"

  exit $?
elif $is_partition; then
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
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
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
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
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
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
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
