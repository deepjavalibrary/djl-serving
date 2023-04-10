#!/usr/bin/env bash

# TODO: change the hack on the code to support A10G
git clone https://github.com/lanking520/AITemplate --recursive
cd AITemplate/python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
cd ../../
rm -rf AITemplate
