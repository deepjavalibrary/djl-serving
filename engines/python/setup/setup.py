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

import setuptools.command.build_py
from setuptools import setup, find_packages

pkgs = find_packages(exclude='src')


def detect_version():
    with open("../../../gradle.properties", "r") as f:
        for line in f:
            if not line.startswith('#'):
                prop = line.split('=')
                if prop[0] == "djl_version":
                    return prop[1].strip()

    return None


def pypi_description():
    with open('PyPiDescription.rst') as df:
        return df.read()


class BuildPy(setuptools.command.build_py.build_py):

    def run(self):
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    version = detect_version()

    requirements = ['psutil', 'packaging', 'wheel']

    test_requirements = [
        'numpy', 'requests', 'Pillow', 'transformers', 'torch'
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
