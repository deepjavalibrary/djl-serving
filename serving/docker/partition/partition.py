#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import subprocess
import logging
import sys
import os
import json
import argparse
import zipfile
import glob

from pathlib import Path
from properties_manager import PropertiesManager

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29761

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'

FILES_TO_EXTRACT = [
    'djl_python/', 'djl_python/deepspeed.py', 'djl_python/inputs.py',
    'djl_python/outputs.py', 'djl_python/pair_list.py',
    'djl_python/np_util.py', 'djl_python/service_loader.py'
]


def get_python_executable():
    python_executable = os.environ.get("PYTHON_EXECUTABLE")
    if python_executable is None:
        python_executable = "python3"

    return python_executable


class PartitionService(object):
    def __init__(self, props_manager):
        self.properties_manager = props_manager
        self.properties = props_manager.properties()
        self.install_requirements_file()
        self.download_model()

    def download_model(self):
        if "s3url" not in self.properties:
            return
        download_dir = os.environ.get("SERVING_DOWNLOAD_DIR",
                                      '/tmp/download/model/')

        s3url = self.properties["s3url"]
        commands = []
        if Path("/opt/djl/bin/s5cmd").is_file():
            if not s3url.endswith("*"):
                if s3url.endswith("/"):
                    s3url = s3url + '*'
                else:
                    s3url = s3url + '/*'

            commands = [
                "/opt/djl/bin/s5cmd",
                "--retry-count",
                "1",
                "sync",
                s3url,
                download_dir
            ]
        else:
            commands = ["aws",
                        "s3",
                        "sync",
                        s3url,
                        download_dir]

        subprocess.run(commands)
        self.properties['model_id'] = download_dir

    def install_requirements_file(self):
        model_dir = self.properties.get("model_dir")
        file = os.path.join(model_dir, 'requirements.txt')
        if os.path.isfile(file):
            command = [
                get_python_executable(), "-m", "pip", "-q", "install", "-r",
                str(file)
            ]
            try:
                result = subprocess.run(command)
                if result.returncode == 0:
                    logging.info("pip install requirements succeed!")
                else:
                    logging.info(
                        f"requirements installation failed! With error: {result}"
                    )
            except Exception as e:
                logging.exception(
                    f"Could not install requirements.txt {str(e)}")

    def get_environmental_vars(self):
        python_path = os.environ.get("PYTHONPATH")
        python_path = f"{python_path},{PYTHON_CACHE_DIR}" if python_path else PYTHON_CACHE_DIR
        python_path += self.properties['model_dir']

    def run_partition(self):
        commands = [
            "mpirun", "-N",
            self.properties.get("tensor_parallel_degree", 1),
            "--allow-run-as-root", "--mca", "btl_vader_single_copy_mechanism",
            "none", "--tag-output", "-x", "FI_PROVIDER=efa", "-x",
            "RDMAV_FORK_SAFE=1", "-x", "FI_EFA_USE_DEVICE_RDMA=1", "-x",
            "LD_LIBRARY_PATH", "-x", f"MASTER_ADDR={MASTER_ADDR}", "-x",
            f"MASTER_PORT={MASTER_PORT}", "-x", "PYTHONPATH",
            get_python_executable(), "run_partition.py", "--properties",
            str(json.dumps(self.properties))
        ]

        result = subprocess.run(commands, env=self.get_environmental_vars())
        logging.info(result)
        if result.returncode == 0:
            logging.info(f"Partitioning done.")
            self.properties_manager.generate_properties_file()
        else:
            raise Exception("Partitioning was not successful.")


def extract_python_jar():
    os.makedirs(PYTHON_CACHE_DIR, exist_ok=True)
    jarfiles = glob.glob('/usr/local/djl-serving-*/lib/python-*.jar')

    with zipfile.ZipFile(jarfiles[0], 'r') as zip:
        # Extracting only required files into a specific location.
        for file in FILES_TO_EXTRACT:
            zip.extract(file, path=PYTHON_CACHE_DIR)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        help='path of the model directory containing model/properties file')

    args = parser.parse_args()

    extract_python_jar()

    properties_manager = PropertiesManager(args.model_dir)
    service = PartitionService(properties_manager)
    service.run_partition()