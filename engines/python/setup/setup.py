#!/usr/bin/env python
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import os
import setuptools.command.build_py
from setuptools import setup, find_packages

pkgs = find_packages(exclude='src')


def detect_version():
    with open("../../../gradle/libs.versions.toml", "r") as f:
        for line in f:
            if not line.startswith('#'):
                prop = line.split('=')
                if prop[0].strip() == "serving":
                    return prop[1].strip().replace('"', '')

    return None


def add_version(version_str):
    djl_version_string = f"__version__ = '{version_str}'"
    with open(os.path.join("djl_python", "__init__.py"), 'r') as f:
        existing = [i.strip() for i in f.readlines()]
    with open(os.path.join("djl_python", "__init__.py"), 'a') as f:
        if djl_version_string not in existing:
            f.writelines(['\n', djl_version_string])


def pypi_description():
    with open('PyPiDescription.rst') as df:
        return df.read()


class BuildPy(setuptools.command.build_py.build_py):

    def run(self):
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    version = detect_version()
    add_version(version)

    requirements = ['psutil', 'packaging', 'wheel']

    test_requirements = [
        'numpy<2', 'requests', 'Pillow', 'transformers', 'torch', 'einops',
        'accelerate', 'sentencepiece', 'protobuf', "peft", 'yapf',
        'pydantic>=2.0', "objgraph"
    ]

    setup(name='djl_python',
          version=version,
          description=
          'djl_python is a tool to build and test DJL Python model locally',
          author='Deep Java Library team',
          author_email='djl-dev@amazon.com',
          long_description=pypi_description(),
          url='https://github.com/deepjavalibrary/djl.git',
          keywords='DJL Serving Deep Learning Inference AI',
          packages=pkgs,
          cmdclass={
              'build_py': BuildPy,
          },
          install_requires=requirements,
          extras_require={'test': test_requirements + requirements},
          entry_points={
              'console_scripts': [
                  'djl-test-model=djl_python.test_model:run',
              ]
          },
          include_package_data=True,
          license='Apache License Version 2.0')
