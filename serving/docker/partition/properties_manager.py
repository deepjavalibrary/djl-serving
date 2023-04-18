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
import logging
import os
import glob
import json
import torch
import requests

# Properties to exclude while generating serving.properties
from utils import is_engine_mpi_mode, get_engine_configs, get_download_dir

EXCLUDE_PROPERTIES = [
    'model_id', 'checkpoint', 's3url', 'save_mp_checkpoint_path', 'model_dir',
    'engine', 'upload_checkpoints_s3url'
]

PARTITION_SUPPORTED_ENGINES = ['DeepSpeed', 'FasterTransformer']

CHUNK_SIZE = 4096  # 4MB chunk size


class PropertiesManager(object):

    def __init__(self, properties_dir):
        self.properties = {}
        self.properties_dir = properties_dir
        self.entry_point_url = None

        self.load_properties()

        self.validate_engine()
        self.is_mpi_mode = is_engine_mpi_mode(self.properties['engine'])

        self.set_and_validate_model_dir()

        if self.is_mpi_mode:
            self.validate_tp_degree()

        self.set_and_validate_entry_point()
        self.set_and_validate_save_mp_checkpoint_path()

    def load_properties(self):
        properties_file = os.path.join(self.properties_dir,
                                       'serving.properties')
        if os.path.exists(properties_file):
            with open(properties_file, 'r') as f:
                for line in f:
                    # ignoring line starting with #
                    if line.startswith("#"):
                        continue
                    key, value = line.strip().split('=', 1)
                    self.properties[key.split(".", 1)[-1]] = value
        else:
            raise FileNotFoundError(
                "serving.properties file does not exist in the path provided.")

    def set_and_validate_model_dir(self):
        if 'model_id' in self.properties and 's3url' in self.properties:
            raise KeyError(
                'Both model_id and s3url cannot be in serving.properties')

        if 'model_dir' in self.properties:
            model_files = glob.glob(
                os.path.join(self.properties['model_dir'], '*.bin'))
            if not model_files:
                raise FileNotFoundError(
                    'No .bin files found in the given option.model_dir')
        elif 'model_id' in self.properties:
            self.properties['model_dir'] = self.properties_dir
        elif 's3url' in self.properties:
            # backward compatible only, should be replaced with model_id
            self.properties['model_dir'] = self.properties_dir
            self.properties['model_id'] = self.properties['s3url']
            self.properties.pop('s3url')
        else:
            model_files = glob.glob(os.path.join(self.properties_dir, '*.bin'))
            if model_files:
                self.properties['model_dir'] = self.properties_dir
            else:
                raise KeyError(
                    'Please specify the option.model_dir or option.model_id or include model files in the model-dir.'
                )

    def validate_and_correct_checkpoints_json(self):
        """
        Removes base_dir from ds_inference_checkpoints.json file.

        DeepSpeed writes base_dir directory, which is the path of checkpoints saved to the file.
        Removing the base_dir since the user's deployment environment could be different from partition environment.
        User can specify base_dir argument in deepspeed.init_inference while using this file.

        :return:
        """
        if self.properties['engine'] == 'DeepSpeed':
            config_file = os.path.join(
                self.properties['save_mp_checkpoint_path'],
                'ds_inference_config.json')
            if not os.path.exists(config_file):
                raise Exception("Checkpoints json file was not generated."
                                "Partition was not successful.")

            configs = {}
            with open(config_file) as f:
                configs = json.load(f)

            if not configs.get('base_dir'):
                return

            configs.pop('base_dir')
            with open(config_file, "w") as f:
                json.dump(configs, f)

    def generate_properties_file(self):
        checkpoint_path = self.properties.get('save_mp_checkpoint_path')
        configs = {
            'engine': self.properties['engine'],
        }

        for key, value in self.properties.items():
            if key not in EXCLUDE_PROPERTIES:
                if key == "entryPoint":
                    entry_point = self.properties.get("entryPoint")
                    if entry_point == "model.py":
                        continue
                    elif self.entry_point_url:
                        configs[f'option.{key}'] = self.entry_point_url
                else:
                    configs[f'option.{key}'] = value

        configs.update(get_engine_configs(self.properties))

        properties_file = os.path.join(checkpoint_path, 'serving.properties')
        with open(properties_file, "w") as f:
            for key, value in configs.items():
                f.write(f"{key}={value}\n")

    def validate_engine(self):
        if 'engine' not in self.properties:
            raise KeyError('Please specify engine in serving.properties')
        elif self.properties['engine'] not in PARTITION_SUPPORTED_ENGINES:
            raise NotImplementedError(
                f'{self.properties["engine"]} '
                f'engine is not supported for ahead of time partitioning.')

    def validate_tp_degree(self):
        if 'tensor_parallel_degree' not in self.properties:
            raise ValueError(
                'Please specify tensor_parallel_degree in serving.properties')

        num_gpus = torch.cuda.device_count()
        tensor_parallel_degree = self.properties['tensor_parallel_degree']
        if num_gpus < int(tensor_parallel_degree):
            raise ValueError(
                f'GPU devices are not enough to run {tensor_parallel_degree} partitions.'
            )

    def set_and_validate_entry_point(self):
        entry_point = self.properties.get('entryPoint')
        if entry_point is None:
            entry_point = os.environ.get("DJL_ENTRY_POINT")
            if entry_point is None:
                entry_point_file = glob.glob(
                    os.path.join(self.properties_dir, 'model.py'))
                if entry_point_file:
                    self.properties['entryPoint'] = 'model.py'
                else:
                    engine = self.properties['engine']
                    if engine == "DeepSpeed":
                        entry_point = "djl_python.deepspeed"
                    elif engine == "FasterTransformer":
                        entry_point = "djl_python.fastertransformer"
                    else:
                        raise FileNotFoundError(
                            f"model.py not found in model path {self.properties_dir}"
                        )
                    self.properties['entryPoint'] = entry_point
        elif entry_point.lower().startswith('http'):
            logging.info(f'Downloading entrypoint file.')
            self.entry_point_url = self.properties['entryPoint']
            download_dir = get_download_dir(self.properties_dir,
                                            suffix='modelfile')
            model_file = os.path.join(download_dir, 'model.py')
            with requests.get(self.properties['entryPoint'], stream=True) as r:
                with open(model_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            self.properties['entryPoint'] = model_file
            logging.info(f'Entrypoint file downloaded successfully')

    def set_and_validate_save_mp_checkpoint_path(self):
        save_mp_checkpoint_path = self.properties["save_mp_checkpoint_path"]
        if save_mp_checkpoint_path.startswith("s3://"):
            self.properties[
                "upload_checkpoints_s3url"] = save_mp_checkpoint_path
            self.properties["save_mp_checkpoint_path"] = get_download_dir(
                self.properties_dir, "partition-model")
