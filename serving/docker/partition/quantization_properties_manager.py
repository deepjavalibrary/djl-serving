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

from properties_manager import PropertiesManager

class QuantizationPropertiesManager(PropertiesManager):

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

