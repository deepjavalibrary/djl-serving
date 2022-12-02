#!/usr/bin/env bash

# https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/logging-and-monitoring.html
TELEMETRY_FILE="/opt/djl/bin/telemetry"
[[ ! -z "${OPT_OUT_TRACKING}" ]] && exit 0
[[ ! -f "${TELEMETRY_FILE}" ]] && exit 0

# IMDS V2
curl -sf -m 1 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 600" >/dev/null 2>&1 || IMDS_V1=true

if [[ "${IMDS_V1}" == true ]]; then
  curl -sf -m 1 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1 || exit 0
  INSTANCE_ID=$(curl -sf -m 1 http://169.254.169.254/latest/meta-data/instance-id)
  REGION=$(curl -sf -m 1 http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')
else
  curl -sf -m 1 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 600" >/dev/null 2>&1 || exit 0
  TOKEN=$(curl -sf -m 1 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
  INSTANCE_ID=$(curl -sf -m 1 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
  REGION=$(curl -sf -m 1 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')
fi

valid_regions="ap-northeast-1 ap-northeast-2 ap-southeast-1 ap-southeast-2 ap-south-1 ca-central-1 eu-central-1 eu-north-1 eu-west-1 eu-west-2 eu-west-3 sa-east-1 us-east-1 us-east-2 us-west-1 us-west-2"
found=0
for region in $valid_regions; do
  if [[ "${region}" == "${REGION}" ]]; then
    found=1
  fi
done

[[ $found == 0 ]] && exit 0
line=$(head -n 1 $TELEMETRY_FILE)
arr=($line)

if [[ ! -z "${TEST_TELEMETRY_COLLECTION}" ]]; then
  echo "${REGION} ${INSTANCE_ID} ${line}" >/opt/djl/logs/telemetry-test
fi

curl -fs -m 1 "https://aws-deep-learning-containers-${REGION}.s3.${REGION}.amazonaws.com/dlc-containers-${INSTANCE_ID}.txt?x-instance-id=${INSTANCE_ID}&x-framework=djl&x-framework_version=${arr[0]}&x-py_version=3&x-container_type=${arr[1]}" >/dev/null 2>&1
