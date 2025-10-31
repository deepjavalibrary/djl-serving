#!/usr/bin/env bash
# used in the dockerfiles to create virtualenvs per engine
# currently only intended for use in lmi.Dockerfile, need to refactor this to work for trtllm if needed
venv_directory=$1
requirements_file=$2

# This was copied over from the previous pip install defined in the lmi.Dockerfile, so it's specific to that Dockerfile
python -m venv --system-site-packages $venv_directory
venv_pip="${venv_directory}/bin/pip"
$venv_pip install -r $requirements_file || exit 1
$venv_pip install https://publish.djl.ai/djl_converter/djl_converter-0.31.0-py3-none-any.whl --no-deps
$venv_pip cache purge
