#!/bin/bash
#set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    code=77
    while [[ code -eq 77 ]]
    do
        /usr/bin/djl-serving "$@"
        code=$?
    done
elif [[ "$1" = "partition" ]]; then
    shift 1
    /usr/bin/python3 partition/partition.py "$@"
else
    eval "$@"
fi
