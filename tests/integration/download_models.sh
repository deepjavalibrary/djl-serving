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
)

python_xgb_models_urls=(
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_ubj_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_deprecated_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_unsafe_model_v2.zip"
  "https://resources.djl.ai/test-models/python/xgboost/xgboost_custom_model_v2.zip"
)

download() {
  urls=("$@")
  for url in "${urls[@]}"; do
    if [[ "$url" == */ ]]; then
      # Directory URL - use wget to download recursively
      dirname=$(basename "${url%/}")
      if ! [ -d "${dirname}" ]; then
        wget -r -np -nH --cut-dirs=3 -R "index.html*" "$url"
      fi
    else
      # File URL - use curl with cache-busting headers
      filename=${url##*/}
      if ! [ -f "${filename}" ]; then
        curl -sf -H "Cache-Control: no-cache" -H "Pragma: no-cache" -O "$url"
      fi
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
