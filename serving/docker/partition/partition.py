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
import glob
import shutil
import logging
import argparse
import subprocess

from pathlib import Path

import utils
from properties_manager import PropertiesManager
from huggingface_hub import snapshot_download

from utils import (get_partition_cmd, extract_python_jar,
                   get_python_executable, get_download_dir,
                   read_hf_model_config, init_hf_tokenizer,
                   remove_option_from_properties)

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
        model_id = self.properties.get("option.model_id")
        if not model_id or not model_id.startswith("s3://"):
            return

        download_dir = os.environ.get(
            "SERVING_DOWNLOAD_DIR",
            get_download_dir(self.properties_manager.properties_dir, 'model'))

        s3url = model_id
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

        self.properties['option.model_id'] = download_dir

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
        if glob.glob(self.properties['option.model_id']):
            return self.properties['option.model_id']

        download_dir = os.environ.get("SERVING_DOWNLOAD_DIR",
                                      '/tmp/download/model/')

        model_name = self.properties['option.model_id']
        downloaded_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=download_dir,
            allow_patterns=CONFIG_FILES_PATTERNS,
        )
        return downloaded_dir

    def copy_config_files(self):
        model_dir = self.properties['model_dir']
        if 'option.model_id' in self.properties:
            model_dir = self.download_config_from_hf()

        config_files = []
        for pattern in CONFIG_FILES_PATTERNS:
            config_files += glob.glob(os.path.join(model_dir, pattern))

        for file in config_files:
            shutil.copy(file,
                        dst=self.properties['option.save_mp_checkpoint_path'])

    def upload_checkpoints_to_s3(self):
        if 'upload_checkpoints_s3url' not in self.properties:
            return

        s3url = self.properties['upload_checkpoints_s3url']
        saved_checkpoints_dir = self.properties[
            "option.save_mp_checkpoint_path"]

        if not saved_checkpoints_dir.endswith('/'):
            saved_checkpoints_dir = saved_checkpoints_dir + '/'

        if not s3url.endswith('/'):
            s3url = s3url + '/'

        if Path("/opt/djl/bin/s5cmd").is_file():
            commands = [
                "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync",
                saved_checkpoints_dir, s3url
            ]
        else:
            commands = ["aws", "s3", "sync", saved_checkpoints_dir, s3url]

        subprocess.run(commands)
        shutil.rmtree(self.properties["option.save_mp_checkpoint_path"])

    def cleanup(self):
        """
        Cleans up the downloaded files in tmp.
        """
        if self.properties_manager.entry_point_url:
            entrypoint_dir = Path(self.properties['entryPoint']).parent
            shutil.rmtree(entrypoint_dir)

    def run_partition(self) -> str:
        """
        :return: the output of the partition command captured from stdout
        """
        commands = get_partition_cmd(self.properties_manager.is_mpi_mode,
                                     self.properties)
        logging.info(f"cmd: {commands}")
        self.set_environmental_vars()
        partition_stdout = ""
        # Use Popen to capture stdout without delaying terminal output
        with subprocess.Popen(commands,
                              stdout=subprocess.PIPE,
                              bufsize=1,
                              universal_newlines=True) as proc:
            for line in proc.stdout:
                partition_stdout += line
                print(line, end='')
        logging.info(proc)
        if proc.returncode == 0:
            logging.info("Partitioning done.")
            self.properties_manager.validate_and_correct_checkpoints_json()
            self.properties_manager.generate_properties_file()
            if not self.properties_manager.skip_copy:
                logging.info("Copying config files...")
                self.copy_config_files()
            self.load_the_generated_checkpoints()
            self.upload_checkpoints_to_s3()
            self.cleanup()
            return partition_stdout
        else:
            raise Exception("Partitioning was not successful.")

    def load_the_generated_checkpoints(self):
        if self.properties['engine'] == 'DeepSpeed':
            saved_checkpoints_dir = self.properties[
                "option.save_mp_checkpoint_path"]
            properties = utils.load_properties(saved_checkpoints_dir)
            if not self.properties_manager.skip_copy:
                properties['model_dir'] = saved_checkpoints_dir
            properties['option.entryPoint'] = self.properties[
                'option.entryPoint']
            properties['partition_handler'] = 'handle'

            entry_point_file = None
            if properties['option.entryPoint'] == 'model.py':
                entry_point_file = os.path.join(
                    self.properties_manager.properties_dir, 'model.py')
                shutil.copy(entry_point_file, saved_checkpoints_dir)

            commands = get_partition_cmd(True, properties)
            self.set_environmental_vars()
            result = subprocess.run(commands)
            logging.info(result)
            if result.returncode == 0:
                logging.info(
                    "Successfully loaded the partitioned checkpoints.")
            else:
                raise Exception("DeepSpeed does not support partitioning. "
                                "Please use a different engine")
            if entry_point_file:
                os.remove(os.path.join(saved_checkpoints_dir, 'model.py'))

    def run_quantization(self):
        quant_method = self.properties['option.quantize']
        if quant_method == 'awq':
            logging.info("Running AutoAWQ quantization")
            self.autoawq_quantize()
            self.properties_manager.generate_properties_file()
            self.upload_checkpoints_to_s3()
        else:
            raise Exception(f"Invalid quantization method: {quant_method}")

    def autoawq_quantize(self):
        """
        Quantizes model using AutoAWQ. Saves output to save_mp_checkpoint_path.
        """
        from awq import AutoAWQForCausalLM
        sys.path.append(PYTHON_CACHE_DIR)
        from djl_python.properties_manager.hf_properties import HuggingFaceProperties

        logging.info(f"Properties: {self.properties}")
        properties = remove_option_from_properties(self.properties)
        hf_configs = HuggingFaceProperties(**properties)
        model_config = read_hf_model_config(hf_configs.model_id_or_path,
                                            hf_configs)
        tokenizer = init_hf_tokenizer(hf_configs.model_id_or_path, hf_configs)

        # Hard-coding these options for now. If vLLM continues to prioritize
        # AutoAWQ we will expose these options to customers in the future.
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
        logging.info(f"Model loading kwargs: {hf_configs.kwargs}")
        awq_model = AutoAWQForCausalLM.from_pretrained(
            hf_configs.model_id_or_path, **hf_configs.kwargs)
        awq_model.quantize(tokenizer, quant_config=quant_config)

        output_path = self.properties.save_mp_checkpoint_path
        logging.info(f"Saving model and tokenizer to: {output_path}")
        awq_model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        required=False,
        default='/opt/ml/input/data/training',
        dest='properties_dir',
        help='path of the model directory containing model/properties file')
    parser.add_argument('--model-id',
                        type=str,
                        required=False,
                        help='HuggingFace model_id or s3_uri')
    parser.add_argument('--engine', type=str, required=False, help='engine')
    parser.add_argument(
        '--save-mp-checkpoint-path',
        type=str,
        required=False,
        help='local path or s3 uri to save/upload the partitioned checkpoints')
    parser.add_argument('--tensor-parallel-degree',
                        type=str,
                        required=False,
                        help='tensor parallel degree')
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help=
        'toggle to skip copying associated tokenizer and config files from source model'
    )
    parser.add_argument('--quantization',
                        type=str,
                        dest='quantize',
                        help="the quantization technique to use. options: awq")

    args = parser.parse_args()

    try:
        properties_manager = PropertiesManager(args)
    except ValueError as e:
        logging.error(str(e))
        parser.print_usage()
        return

    extract_python_jar(PYTHON_CACHE_DIR)

    service = PartitionService(properties_manager)
    if properties_manager.properties.get('option.quantize'):
        service.run_quantization()
    else:
        service.run_partition()


if __name__ == "__main__":
    main()
