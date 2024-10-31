#!/bin/bash
#set -e

if [[ -n "$SM_NEO_EXECUTION_CONTEXT" ]]; then
  echo "SageMaker Neo execution context detected"
  /usr/bin/python3 /opt/djl/partition/sm_neo_dispatcher.py
  exit $?
fi

if [[ "$1" = "serve" ]]; then
  shift 1
  code=77
  while [[ code -eq 77 ]]; do
    /usr/bin/djl-serving "$@"
    code=$?
  done
  exit $code
elif [[ "$1" = "partition" ]] || [[ "$1" = "train" ]]; then
  set -e
  shift 1
  /usr/bin/python3 /opt/djl/partition/partition.py "$@"
else
  set -e
  eval "$@"
fi
