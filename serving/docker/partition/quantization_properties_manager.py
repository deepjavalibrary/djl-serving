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
import torch

from properties_manager import PropertiesManager

SUPPORTED_QUANTIZATION_METHODS = ["awq", "fp8"]

class QuantizationPropertiesManager(PropertiesManager):
    def __init__(self, args, **kwargs):
        super.__init__(args, **kwargs)

        if args.quantize:
            self.properties['option.quantize'] = args.quantize

        self.validate_quantization_method()


    def validate_tp_degree(self):
        """
        Validates tensor_parallel_degree, defaulting to 
        tensor_parallel_degree = 1. Overrides the parent class function,
        which raises an exception if tensor_parallel_degree is not specified.
        """
        tensor_parallel_degree = self.properties.get(
            'option.tensor_parallel_degree')
        if not tensor_parallel_degree:
            self.properties['option.tensor_parallel_degree'] = 1

        num_gpus = torch.cuda.device_count()
        if num_gpus < int(tensor_parallel_degree):
            raise ValueError(
                f'GPU devices are not enough to run {tensor_parallel_degree} partitions.'
            )


    def set_and_validate_entry_point(self):
        entry_point = self.properties.get('option.entryPoint')
        if entry_point is None:
            entry_point = os.environ.get("DJL_ENTRY_POINT")
            if entry_point is None:
                engine = self.properties.get('engine')
                if engine.lower() == "mpi":
                    self.properties['option.entryPoint'] = "djl_python.huggingface"
                    self.properties['option.mpi_mode'] = "True"
                else:
                    raise ValueError(f"Invalid engine: {engine}. Quantization only supports MPI engine.")
        elif entry_point != "djl_python.huggingface":
            raise ValueError(f"Invalid entrypoint for Quantization: {entry_point}")


    def validate_quantization_method(self):
        quantize = self.properties.get('option.quantize')

        if quantize:
            if quantize not in SUPPORTED_QUANTIZATION_METHODS:
                raise ValueError(f"Quantize method: {quantize} not supported. Support options are: {SUPPORTED_QUANTIZATION_METHODS}")
        else:
            raise ValueError(f"Quantize method not specified. Options are: {SUPPORTED_QUANTIZATION_METHODS}")

