#!/bin/bash

rm -rf flash-attention
git clone https://github.com/HazyResearch/flash-attention.git -b v1.0.0
pushd flash-attention || exit 1
python setup.py install
cd csrc/layer_norm && python setup.py install
cd ../rotary && python setup.py install
popd || exit 1
rm -rf flash-attention
