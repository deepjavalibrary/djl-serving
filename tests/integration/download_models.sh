#!/bin/bash

set -e

platform=$1 # expected values are "cpu" "cpu-full" "pytorch-gpu" "pytorch-inf2" "aarch64"

rm -rf models
mkdir models && cd models
curl -sf -O https://resources.djl.ai/images/kitten.jpg

# all platform models except aarch
general_platform_models_urls=(
  "https://resources.djl.ai/test-models/pytorch/resnet18_all_batch.zip"
  "https://resources.djl.ai/test-models/tensorflow/resnet50v1.zip"
  "https://resources.djl.ai/test-models/onnxruntime/resnet18-v1-7.zip"
)

# only pytorch and onnx models
aarch_models_urls=(
  "https://resources.djl.ai/test-models/pytorch/resnet18_all_batch.zip"
  "https://resources.djl.ai/test-models/onnxruntime/resnet18-v1-7.zip"
)

inf2_models_urls=(
  "https://resources.djl.ai/test-models/pytorch/resnet18_inf2_2_4.tar.gz"
  "https://resources.djl.ai/test-models/pytorch/resnet18_no_reqs_inf2_2_4.tar.gz"
)

python_skl_models_urls=(
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_joblib_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_cloudpickle_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_skops_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_multi_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_unsafe_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_custom_model_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_skops_model_env_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_custom_model_sm_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_custom_model_input_output_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_custom_model_input_output_invalid_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_mixed_djl_sagemaker_v2.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_djl_all_formatters_v4.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_djl_input_output_v3.zip"
  "https://resources.djl.ai/test-models/python/sklearn/sklearn_djl_invalid_input_v3.zip"
  "https://resources.djl.ai/test-models/python/sklearn/slow_loading_model.zip"
  "https://resources.djl.ai/test-models/python/sklearn/slow_predict_model.zip"
)

python_xgb_models_urls=(
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_ubj_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_deprecated_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_unsafe_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_custom_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_sagemaker_all.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_sagemaker_input_output.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_sagemaker_input_output_invalid.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_mixed_djl_sagemaker_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_djl_all_formatters.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_djl_input_output_v3.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_djl_invalid_input_v3.zip"
)

download() {
  urls=("$@")
  for url in "${urls[@]}"; do
    filename=${url##*/}
    # does not download the file, if file already exists
    if ! [ -f "${filename}" ]; then
      curl -sf -O "$url"
    fi
  done
}

case $platform in
cpu | pytorch-gpu)
  download "${general_platform_models_urls[@]}"
  ;;
cpu-full)
  download "${general_platform_models_urls[@]}"
  download "${python_skl_models_urls[@]}"
  download "${python_xgb_models_urls[@]}"
  ;;
pytorch-inf2)
  download "${inf2_models_urls[@]}"
  ;;
aarch64)
  download "${aarch_models_urls[@]}"
  ;;
*)
  echo "Bad argument. Expecting one of the values: cpu, cpu-full, pytorch-gpu, pytorch-inf2, aarch64"
  exit 1
  ;;
esac
