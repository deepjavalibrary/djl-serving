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
import logging
import subprocess
import sys
import argparse

from utils import extract_python_jar, get_quantization_cmd
from partition import PartitionService, PYTHON_CACHE_DIR
from quantization_properties_manager import QuantizationPropertiesManager


class QuantizationService(PartitionService):
    def run_quantization(self) -> str:
        commands = get_quantization_cmd(self.properties)
        logging.info(f"quantize cmd: {commands}")
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
            logging.info("Quantization done.")
            self.properties_manager.generate_properties_file()
            if not self.properties_manager.skip_copy:
                logging.info("Copying config files...")
                self.copy_config_files()
            self.upload_checkpoints_to_s3()
            self.cleanup()
            return partition_stdout
        else:
            raise Exception("Quantization was not successful.")

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

    args = parser.parse_args()

    try:
        properties_manager = QuantizationPropertiesManager(args)
    except ValueError as e:
        logging.error(str(e))
        parser.print_usage()
        return

    extract_python_jar(PYTHON_CACHE_DIR)

    service = QuantizationService(properties_manager)
    service.run_quantization()

if __name__ == "__main__":
    main()
    

