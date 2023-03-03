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
import sys
import os
import json
import glob
import shutil
import zipfile
import logging
import argparse
import subprocess

from pathlib import Path
from properties_manager import PropertiesManager
from huggingface_hub import snapshot_download

from utils import get_partition_cmd, extract_python_jar

PYTHON_CACHE_DIR = '/tmp/djlserving/cache'

CONFIG_FILES_PATTERNS = ["*.json", "*.txt", "*.model"]

ALLOW_PATTERNS = ["*.json", "*.pt", "*.bin", "*.txt"]


class PartitionService(object):
    def __init__(self, props_manager):
        self.properties_manager = props_manager
        self.properties = props_manager.properties
        self.install_requirements_file()
        self.download_model_from_s3()

    def download_model_from_s3(self):
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
                "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync", s3url,
                download_dir
            ]
        else:
            commands = ["aws", "s3", "sync", s3url, download_dir]

        subprocess.run(commands)

        # check if any file was downloaded.
        if not glob.glob(os.path.join(download_dir, '*')):
            raise Exception('Model download from s3url failed')

        self.properties['model_id'] = download_dir

    def install_requirements_file(self):
        req_file_dir = self.properties_manager.properties_dir
        file = os.path.join(req_file_dir, 'requirements.txt')
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

    def set_environmental_vars(self):
        environments = {}
        python_path = []
        if os.environ.get("PYTHONPATH"):
            python_path.append(os.environ.get("PYTHONPATH"))
        python_path.append(PYTHON_CACHE_DIR)
        if 'model_dir' in self.properties:
            python_path.append(self.properties['model_dir'])
        environments['PYTHONPATH'] = ':'.join(python_path)
        os.environ.update(environments)

    def download_config_from_hf(self):
        # checks if model_id is a path
        if glob.glob(self.properties['model_id']):
            return self.properties['model_id']

        download_dir = os.environ.get("SERVING_DOWNLOAD_DIR",
                                      '/tmp/download/model/')

        model_name = self.properties['model_id']
        downloaded_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=download_dir,
            allow_patterns=CONFIG_FILES_PATTERNS,
        )
        return downloaded_dir

    def copy_config_files(self):
        model_dir = self.properties['model_dir']
        if 'model_id' in self.properties:
            model_dir = self.download_config_from_hf()

        config_files = []
        for pattern in CONFIG_FILES_PATTERNS:
            config_files += glob.glob(os.path.join(model_dir, pattern))

        for file in config_files:
            shutil.copy(file, dst=self.properties['save_mp_checkpoint_path'])

    def run_partition(self):
        commands = get_partition_cmd(self.properties_manager.is_mpi_mode,
                                     self.properties)
        self.set_environmental_vars()
        result = subprocess.run(commands)
        logging.info(result)
        if result.returncode == 0:
            logging.info(f"Partitioning done.")
            self.properties_manager.generate_properties_file()
            self.copy_config_files()
        else:
            raise Exception("Partitioning was not successful.")


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

    extract_python_jar(PYTHON_CACHE_DIR)

    properties_manager = PropertiesManager(args.model_dir)
    service = PartitionService(properties_manager)
    service.run_partition()
