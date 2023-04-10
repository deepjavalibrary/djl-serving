#!/bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
	case $1 in
		--docker_image)
			DOCKER_IMAGE="$2"
			shift
			shift;;
		--profile)
			PROFILE="$2"
			shift
			shift;;
		--engine)
			ENGINE="$2"
			shift
			shift;;
		-h|--handler)
			HANDLER="$2"
			shift
			shift;;
		-m|--model)
			MODEL="$2"
			shift
			shift;;
		--dtype)
			DTYPE="$2"
			shift
			shift;;
		--batch_size)
			BATCH_SIZE="$2"
			shift
			shift;;
		--in_tokens)
			IN_TOKENS="$2"
			shift
			shift;;
		--out_tokens)
			OUT_TOKENS="$2"
			shift
			shift;;
		--tensor_parallel)
			TENSOR_PARALLEL="$2"
			shift
			shift;;
		-c|--count)
			COUNT="$2"
			shift
			shift;;
		-l|--log_metrics)
			LOG_METRICS="true"
			shift;;
		-*)
			echo "Unknown option $1"
			exit 1;;
		*)
			POSITIONAL_ARGS+=("$1")
			shift;;
	esac
done

set -- "${POSITIONAL_ARGS[@]}"

# =================== Validate Command Line Arguments ===========================
if [[ -z $HANDLER || -z $ENGINE || -z $DOCKER_IMAGE || -z $MODEL || -z $DTYPE || -z $TENSOR_PARALLEL ]] || [[ -z $PROFILE || -z $DOCKER_IMAGE ]]; then
  echo "Missing mandatory arguments: [engine, docker_image, model, dtype, tensor_parallel]"
  echo "Usage: ./lmic_test.sh --engine [engine] --docker_image [docker_image] ..."
fi

if [[ -z $PROFILE ]]; then
	ENGINES=("$ENGINE")
	DTYPES=("$DTYPE")
	IN_TOKENS=("$IN_TOKENS")
fi

# =================== Profile Argument Parsing ====================
if [[ -n $PROFILE ]]; then
  # ======== Required Profile Single Argument Parameters ==========
	HANDLER=$( < "$PROFILE" jq -r '.handler | @sh' | tr -d \')
	MODEL=$( < "$PROFILE" jq -r '.model | @sh' | tr -d \')
	TENSOR_PARALLEL=$( < "$PROFILE" jq -r '.tensor_parallel | @sh' | tr -d \')
	BATCH_SIZE=$( < "$PROFILE" jq -r '.batch_size | @sh' | tr -d \')
	OUT_TOKENS=$( < "$PROFILE" jq -r '.out_tokens | @sh' | tr -d \')
	COUNT=$( < "$PROFILE" jq -r '.count | @sh' | tr -d \')

  # ======== Required Profile Single or Array Argument Parameters ==========
	IFS=" " read -r -a ENGINES <<< "$( < "$PROFILE" jq -r '.engine | @sh' | tr -d \')"
	IFS=" " read -r -a DTYPES <<< "$( < "$PROFILE" jq -r '.dtype | @sh' | tr -d \')"

	# ======= Optional Profile Single Argument Parameters ============================
	if [[ $( < "$PROFILE" jq -r '.log_metrics | @sh' | tr -d \') != "null" ]]; then
		LOG_METRICS="true"
	fi

	# ======= Optional Profile Single or Array Argument Parameters ============================
	if [[ $( < "$PROFILE" jq -r '.in_tokens | @sh' | tr -d \') != "null" ]]; then
	  IFS=" " read -r -a IN_TOKENS <<< "$( < "$PROFILE" jq -r '.in_tokens | @sh' | tr -d \')"
	fi
fi

# ======================Testing Script==============================

docker pull "$DOCKER_IMAGE"

for ENGINE in "${ENGINES[@]}"; do
	for DTYPE in "${DTYPES[@]}"; do
    rm -rf models
    # The reason for the sed command below is because we need to account for huggingface model_id's which have a /
    PREFIX=$(echo "${ENGINE}_${MODEL}_${DTYPE}_${BATCH_SIZE}_${TENSOR_PARALLEL}" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    METRICS="${PREFIX}.log"
    CPU="${PREFIX}_cpu.log"

    nohup ./cpu_memory_monitor.sh > "$CPU" 2>&1 &
    mem_pid=$!

    python3 llm/prepare.py "$HANDLER" "$MODEL" \
      --engine "$ENGINE" \
      --dtype "$DTYPE" \
      --tensor_parallel "$TENSOR_PARALLEL"

    ./launch_container.sh "$DOCKER_IMAGE" "$PWD"/models lmic_performance \
          serve -m test=file:/opt/ml/model/test/

    kill $mem_pid
    MAX=$(awk 'BEGIN{a=0}{if ($2>0+a) a=$2} END{print a}' "$CPU")
    MIN=$(awk -v var="$MAX" 'BEGIN{a=var}{if ($2<0+a) a=$2} END{print a}' "$CPU")
    CPU_USAGE=$((MAX - MIN))

    for IN_TOKEN in "${IN_TOKENS[@]}"; do
      python3 -u llm/client.py "$HANDLER" "$MODEL" \
          --engine "$ENGINE" \
          --dtype "$DTYPE" \
          --tensor_parallel "$TENSOR_PARALLEL" \
          --cpu_mem "$CPU_USAGE" \
          ${BATCH_SIZE:+--batch_size "${BATCH_SIZE}"} \
          ${IN_TOKEN:+--in_tokens "${IN_TOKEN}"} \
          ${OUT_TOKENS:+--out_tokens "${OUT_TOKENS}"} \
          ${COUNT:+--count "${COUNT}"} \
          >> "$METRICS"

      if [[ -n $LOG_METRICS ]]; then
        aws cloudwatch put-metric-data --namespace "LMIC_performance_${ENGINE}" --region "us-east-1" \
            --metric-data "$(sed "s/'/\"/g" "$METRICS")"
      fi
      rm "$METRICS"
    done
    rm "$CPU"
    docker rm -f "$(docker ps -aq)"
	done
done

