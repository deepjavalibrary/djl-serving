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
import os
import glob
import torch

# Properties to exclude while generating serving.properties
from utils import is_engine_mpi_mode, get_engine_configs, get_download_dir

EXCLUDE_PROPERTIES = [
    'model_id', 'checkpoint', 's3url', 'save_mp_checkpoint_path', 'model_dir',
    'engine', 'upload_checkpoints_s3url'
]

PARTITION_SUPPORTED_ENGINES = ['DeepSpeed', 'FasterTransformer']


class PropertiesManager(object):

    def __init__(self, properties_dir):
        self.properties = {}
        self.properties_dir = properties_dir

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

    def generate_properties_file(self):
        checkpoint_path = self.properties.get('save_mp_checkpoint_path')
        configs = {
            'engine': self.properties['engine'],
        }

        for key, value in self.properties.items():
            if key not in EXCLUDE_PROPERTIES:
                if key == "entryPoint" and self.properties.get(
                        "entryPoint") == "model.py":
                    continue
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
        if "entryPoint" not in self.properties:
            entry_point = os.environ.get("DJL_ENTRY_POINT")
            if entry_point is None:
                entry_point_file = glob.glob(
                    os.path.join(self.properties_dir, 'model.py'))
                if entry_point_file:
                    self.properties['entryPoint'] = 'model.py'
                else:
                    engine = self.properties['engine']
                    if engine == "Deepspeed":
                        entry_point = "djl_python.deepspeed"
                    elif engine == "FasterTransformer":
                        entry_point = "djl_python.fastertransformer"
                    else:
                        raise FileNotFoundError(
                            f"model.py not found in model path {self.properties_dir}"
                        )
                    self.properties['entryPoint'] = entry_point

    def set_and_validate_save_mp_checkpoint_path(self):
        save_mp_checkpoint_path = self.properties["save_mp_checkpoint_path"]
        if save_mp_checkpoint_path.startswith("s3://"):
            self.properties[
                "upload_checkpoints_s3url"] = save_mp_checkpoint_path
            self.properties["save_mp_checkpoint_path"] = get_download_dir(
                self.properties_dir, "partition-model")
