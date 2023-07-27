#!/bin/bash

rm -rf flash-attention-v2
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention-v2 -b v2.0.0
pushd flash-attention-v2 || exit 1
pip install -v .
popd || exit 1
rm -rf flash-attention-v2
