#!/bin/bash

model_path=$1

touch $model_path/errors.json
mkdir -p $model_path/cache $model_path/compiled

echo -e "SM_NEO_EXECUTION_CONTEXT=1" > docker_env
echo -e "SM_NEO_INPUT_MODEL_DIR=/opt/ml/model/input" > docker_env
echo -e "SM_NEO_COMPILED_MODEL_DIR=/opt/ml/model/compiled" > docker_env
echo -e "SM_NEO_COMPILATION_ERROR_FILE=/opt/ml/compilation/errors/errors.json" > docker_env
echo -e "SM_NEO_CACHE_DIR=/opt/ml/compilation/cache" > docker_env
echo -e "COMPILER_OPTIONS={}" > docker_env
