#!/bin/bash

LOCATION=$1

if ! [ -f "${LOCATION}" ]; then
  echo "telemetry file ${LOCATION} not found!"
  exit 1;
fi

result=($(head -n 1 $LOCATION))

if [[ "${result[0]}" != "us-east-1" ]]; then
  echo "region ${result[0]} is not correct!"
  exit 1;
fi

if ! [[ "${result[1]}" =~ ^i\-[0-9a-z]+$ ]]; then
  echo "instance id ${result[1]} is not correct!"
  exit 1;
fi

if ! [[ "${result[2]}" =~ ^[0-9]\.[0-9]+\.[0-9].*$ ]]; then
  echo "version ${result[2]} is not correct!"
  exit 1;
fi
