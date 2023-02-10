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

# Properties to exclude while generating serving.properties
EXCLUDE_PROPERTIES = ['model_id',
                      'checkpoint',
                      's3url',
                      'save_mp_checkpoint_path',
                      'model_dir']

PARTITION_SUPPORTED_ENGINES = ['DeepSpeed']


class PropertiesManager(object):

    def __init__(self, model_dir):
        self.properties = []
        self.load_properties(model_dir)
        self.validate_model()
        self.validate_engine()
        self.validate_tp_degree()

    def load_properties(self, properties_dir):
        properties_file = os.path.join(properties_dir, 'serving.properties')
        if os.path.exists(properties_file):
            with open(properties_file, 'r') as f:
                for line in f:
                    # ignoring line starting with #
                    if line.startswith("#"):
                        continue
                    key, value = line.strip().split('=', 1)
                    if key.startswith("option"):
                        self.properties[key[7:]] = value
                    else:
                        self.properties[key] = value
        else:
            raise Exception("serving.properties file does not exist in the path provided.")

    def generate_properties_file(self):
        checkpoint_path = self.properties.get('save_mp_checkpoint_path')

        checkpoint_json = os.path.join(checkpoint_path, 'ds_inference_config.json')
        if not os.path.exists(checkpoint_json):
            raise Exception('Partition was not successful')

        configs = {
            'option.model_dir': checkpoint_path,
            'option.checkpoint': 'ds_inference_config.json',
        }

        for key, value in self.properties.items():
            if key not in EXCLUDE_PROPERTIES:
                configs[f'option.{key}'] = value

        properties_file = os.path.join(checkpoint_path, 'serving.properties')
        with open(properties_file, "w") as f:
            for key, value in configs.items():
                f.write(f"{key}={value}\n")

    def validate_engine(self):
        if 'engine' not in self.properties:
            raise Exception('Please specify engine in serving.properties')
        elif self.properties['engine'] not in PARTITION_SUPPORTED_ENGINES:
            raise Exception(f'{self.properties["engine"]} engine is not supported for ahead of time partitioning.')

    def validate_tp_degree(self):
        if 'tensor_parallel_degree' not in self.properties:
            raise Exception('Please specify tensor_parallel_degree in serving.properties')
        # TODO: get the GPU device and validate.

    def validate_model(self):
        if 'model_id' in self.properties:
            return
        if 'model_dir' in self.properties:
            model_files = glob.glob(os.path.join(self.properties, '*.bin'))
            if not model_files:
                raise Exception('No .bin files found in the model directory.')