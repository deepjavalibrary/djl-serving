#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sys
import os
from enum import Enum

from sm_neo_utils import get_neo_env_vars
from utils import update_kwargs_with_env_vars, load_properties

VALID_LOAD_FORMATS = ["sagemaker_fast_model_loader"]

# Paths to each Python executable
LMI_DIST_VENV_EXEC = "/opt/djl/lmi_dist_venv/bin/python"
VLLM_VENV_EXEC = "/opt/djl/vllm_venv/bin/python"
SYSTEM_PY_EXEC = "/usr/bin/python3"


class NeoTask(Enum):
    """
    Enum representing available Neo optimization tasks, including their names and script paths
    """

    TENSORRT_LLM = ("TensorRT-LLM compilation",
                    "/opt/djl/partition/sm_neo_trt_llm_partition.py")
    NEURON = ("Neuron compilation",
              "/opt/djl/partition/sm_neo_neuron_partition.py")
    QUANTIZATION = ("Quantization", "/opt/djl/partition/sm_neo_quantize.py")
    SHARDING = ("SageMaker Fast Model Loader sharding",
                "/opt/djl/partition/sm_neo_shard.py")

    def __init__(self, task_name: str, script_path: str):
        self.task_name = task_name
        self.script_path = script_path


class NeoDispatcher:

    def __init__(self):
        self.serving_features = os.environ.get("SERVING_FEATURES")
        self.properties = self._get_serving_properties()

    def _get_serving_properties(self) -> dict:
        neo_environ = get_neo_env_vars()
        properties = update_kwargs_with_env_vars({})
        properties.update(
            load_properties(neo_environ["SM_NEO_INPUT_MODEL_DIR"]))
        return properties

    def is_valid_sharding_config(self):
        if self.properties.get("option.load_format") in VALID_LOAD_FORMATS:
            if self.properties.get("option.quantize"):
                raise ValueError(
                    f"Sharding & Quantization are mutually exclusive. "
                    f"Please Quantize first & then Shard. "
                    f"Received load_format={self.properties.get('option.load_format')} "
                    f"& quantize={self.properties.get('option.quantize')}.")
            if not int(
                    self.properties.get("option.tensor_parallel_degree", "0")):
                raise ValueError(
                    f"Please specify a non-zero tensor_parallel_degree while Sharding. "
                    f"Received {self.properties.get('option.tensor_parallel_degree')}."
                )
            if int(self.properties.get("option.pipeline_parallel_degree",
                                       1)) > 1:
                raise ValueError(
                    f"Sharding does not currently support Pipeline Parallelism. "
                    f"Received {self.properties.get('option.pipeline_parallel_degree')}."
                )
            return True
        return False

    def _get_mpirun_command(self, task: NeoTask, num_processes: int,
                            python_exec: str):
        return [
            "mpirun", "--allow-run-as-root", "--bind-to", "none", "--mca",
            "btl_vader_single_copy_mechanism", "none", "--tag-output", "-x",
            "FI_PROVIDER=efa", "-x", "RDMAV_FORK_SAFE=1", "-x",
            "FI_EFA_USE_DEVICE_RDMA=1", "-x", "LD_LIBRARY_PATH", "-x",
            "PYTHONPATH", "-x", "MKL_DYNAMIC=FALSE", "-np",
            str(num_processes), python_exec, task.script_path
        ]

    def run_task(self, task: NeoTask, python_exec: str):
        """
        Run the specified task as a subprocess, while forwarding its output to the console.
        For sharding jobs, use mpirun to launch multiple processes.
        """
        try:
            if task == NeoTask.SHARDING:
                tp_degree = self.properties.get(
                    "option.tensor_parallel_degree", 1)
                pp_degree = self.properties.get(
                    "option.pipeline_parallel_degree", 1)
                world_size = int(tp_degree) * int(pp_degree)
                cmd = self._get_mpirun_command(task, world_size, python_exec)
            else:
                cmd = [python_exec, task.script_path]

            with subprocess.Popen(cmd,
                                  env=os.environ,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  bufsize=1,
                                  universal_newlines=True) as proc:
                for line in proc.stdout:
                    print(line, end='')

            print(f"{task.task_name} exited with code {proc.returncode}")
            sys.exit(proc.returncode)

        except subprocess.CalledProcessError as e:
            print(f"{task.task_name} exited with code {e.returncode}")
            sys.exit(e.returncode)

    def dispatch(self):
        """
        Determine which Neo task to launch, based on the container's features and the
        present serving properties.
        """
        match self.serving_features:
            case "vllm,lmi-dist":
                if self.is_valid_sharding_config():
                    if self.properties.get("option.rolling_batch",
                                           "lmi-dist").lower() == "vllm":
                        python_exec = VLLM_VENV_EXEC
                    else:
                        python_exec = LMI_DIST_VENV_EXEC
                    print("Sharding Model...")
                    self.run_task(NeoTask.SHARDING, python_exec)
                else:
                    if self.properties.get("option.quantize",
                                           "").lower() == "fp8":
                        python_exec = VLLM_VENV_EXEC
                    else:
                        # run awq quantization with lmi-dist venv b/c AutoAWQ
                        # is incompatible with newer transformers
                        python_exec = LMI_DIST_VENV_EXEC
                    self.run_task(NeoTask.QUANTIZATION, python_exec)
            case "trtllm":
                self.run_task(NeoTask.TENSORRT_LLM, SYSTEM_PY_EXEC)
            case "vllm,lmi-dist,tnx":
                self.run_task(NeoTask.NEURON, SYSTEM_PY_EXEC)
            case _:
                raise ValueError(
                    "Container does not support SageMaker Neo context")


def main():
    dispatcher = NeoDispatcher()
    dispatcher.dispatch()


if __name__ == "__main__":
    main()
