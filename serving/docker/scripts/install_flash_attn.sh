#!/bin/bash

rm -rf flash-attention
git clone https://github.com/HazyResearch/flash-attention.git -b v1.0.0
pushd flash-attention || exit 1
pip install -v .
cd csrc/layer_norm && pip install -v .
cd ../rotary && pip install -v .
popd || exit 1
rm -rf flash-attention
