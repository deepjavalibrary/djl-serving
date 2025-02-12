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
import json
import os
import shutil
import sys
import logging
from importlib.metadata import version
from typing import Final, Optional

from sm_neo_utils import (OptimizationFatalError, write_error_to_file,
                          get_neo_env_vars)
from utils import (update_kwargs_with_env_vars, load_properties)

import torch
from mpi4py import MPI

from lmi_dist.init_engine import engine_from_args
from lmi_dist.arg_utils import VllmEngineArgs
from lmi_dist.comms import comms
from lmi_dist.vllm_engine import load_model_for_sharding

CHUNK_MB = 8


class NeoShardingService():

    def __init__(self):
        neo_environ = get_neo_env_vars()
        self.INPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_INPUT_MODEL_DIR"]
        self.OUTPUT_MODEL_DIRECTORY: Final[str] = neo_environ[
            "SM_NEO_COMPILED_MODEL_DIR"]
        self.COMPILATION_ERROR_FILE: Final[str] = neo_environ[
            "SM_NEO_COMPILATION_ERROR_FILE"]

        self.properties: dict = update_kwargs_with_env_vars({})
        self.properties.update(load_properties(self.INPUT_MODEL_DIRECTORY))
        import sagemaker_fast_model_loader_rust as sm_fml
        py_version = "{}.{}.{}".format(*sys.version_info[:3])

        self.pp_degree = int(
            self.properties.get("option.pipeline_parallel_degree", 1))
        self.tp_degree = int(self.properties["option.tensor_parallel_degree"])
        self.shard_config = sm_fml.ModelConfig(
            pipeline_parallel_size=self.pp_degree,
            tensor_parallel_size=self.tp_degree,
            framework=sm_fml.ModelFramework.Vllm,
            framework_version=version("vllm"),
            python_version=py_version,
        )

    def add_shard_configs(
        self,
        partial_configs: list,
    ):
        for entry in partial_configs:
            if not entry["config"]:
                continue
            self.shard_config.add_shard(
                pipeline_parallel_degree=int(entry["pp"]),
                tensor_parallel_degree=int(entry["tp"]),
                shard_config=entry["config"],
            )

    def save_configs(self, input_dir: str = "", output_dir: str = "") -> None:
        self.shard_config.save(output_dir=output_dir)
        logging.info(
            f"SageMaker Fast Model Loader config file saved to {output_dir}")
        self.copy_non_safetensors_files(input_dir, output_dir)
        logging.info(f"Other non-Safetensors files copied to {output_dir}")

    def copy_non_safetensors_files(self, input_dir: str, output_dir: str):
        """
        Copy all files that are not Safetensors weights from input dir to output dir
        """
        index_json_path = os.path.join(input_dir,
                                       "model.safetensors.index.json")
        if os.path.exists(index_json_path):
            with open(index_json_path, "r") as f:
                index_data = json.load(f)
            safetensors_files = list(index_data["weight_map"].values())
        else:
            # If the index file doesn't exist, assume there is only a single model.safetensors file
            safetensors_files = ["model.safetensors"]

        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if item not in safetensors_files and item != "model.safetensors.index.json":
                if os.path.isfile(item_path):
                    shutil.copy2(item_path, os.path.join(output_dir, item))
                elif os.path.isdir(item_path):
                    shutil.copytree(item_path, os.path.join(output_dir, item))

    def generate_properties_file(self):
        with open(
                os.path.join(self.OUTPUT_MODEL_DIRECTORY,
                             "serving.properties"), "w") as f:
            for key, value in self.properties.items():
                f.write(f"{key}={value}\n")

    # By setting pp_rank and tp_rank_interval , only workers in those ranks will load the model
    # i.e. in case pp=2, tp=4, the arg of pp_rank=1, tp_interval = [2,3,4]
    #       only workers with rank 5, 6, 7 load the model
    def shard_lmi_dist_model(self, input_dir: str, output_dir: str,
                             pp_degree: int, tp_degree: int, chunk_mb: int,
                             target_pp_rank: int,
                             target_tp_rank_interval) -> None:
        # For engine args which can affect GPU memory utilization, use LMI defaults
        # unless specified otherwise by the customer
        gpu_memory_utilization = float(
            self.properties.get("option.gpu_memory_utilization", 0.9))
        enforce_eager: bool = self.properties.get("option.enforce_eager",
                                                  "false").lower() == "true"
        max_rolling_batch_size = int(
            self.properties.get("option.max_rolling_batch_size", 256))
        max_model_len = self.properties.get("option.max_model_len", None)
        if max_model_len is not None:
            max_model_len = int(max_model_len)

        # LoraConfigs
        lora_kwargs = {}
        if enable_lora := self.properties.get("option.enable_lora"):
            enable_lora_bool = enable_lora.lower() == "true"

            if enable_lora_bool:
                max_loras: int = int(
                    self.properties.get("option.max_loras", "4"))
                max_lora_rank: int = int(
                    self.properties.get("option.max_lora_rank", "16"))
                fully_sharded_loras: bool = str(
                    self.properties.get("option.fully_sharded_loras",
                                        "false")).lower() == "true"
                lora_extra_vocab_size: int = int(
                    self.properties.get("option.lora_extra_vocab_size", "256"))
                lora_dtype: str = self.properties.get("option.lora_dtype",
                                                      "auto")
                max_cpu_loras: Optional[int] = None
                if cpu_loras := self.properties.get("option.max_cpu_loras"):
                    max_cpu_loras = int(cpu_loras)

                lora_kwargs["enable_lora"] = enable_lora_bool
                lora_kwargs["fully_sharded_loras"] = fully_sharded_loras
                lora_kwargs["max_loras"] = max_loras
                lora_kwargs["max_lora_rank"] = max_lora_rank
                lora_kwargs["lora_extra_vocab_size"] = lora_extra_vocab_size
                lora_kwargs["lora_dtype"] = lora_dtype
                lora_kwargs["max_cpu_loras"] = max_cpu_loras

        engine_args = VllmEngineArgs(
            model=input_dir,
            pipeline_parallel_size=pp_degree,
            tensor_parallel_size=tp_degree,
            disable_custom_all_reduce=True,
            distributed_executor_backend="mp",
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_rolling_batch_size,
            max_model_len=max_model_len,
            **lora_kwargs,
        )

        engine_configs = engine_args.create_engine_configs()
        engine_worker = load_model_for_sharding(engine_configs, target_pp_rank,
                                                target_tp_rank_interval)

        # Lazy import to avoid MPI not-inited errors
        import sagemaker_fast_model_loader_rust as sm_fml
        model_dir = os.path.join(output_dir, sm_fml.MODEL_DIR_NAME)
        os.makedirs(model_dir, exist_ok=True)

        config_for_current_rank = engine_worker.save_chunked_shard(
            output_dir=model_dir,
            chunk_mb=chunk_mb,
            target_pp_rank=target_pp_rank,
            target_tp_rank_interval=target_tp_rank_interval)

        # Gather results from all ranks to driver process
        configs = MPI.COMM_WORLD.gather(config_for_current_rank, root=0)

        # Driver process saves configs of current rank to disk
        if comms.rank == 0:
            self.add_shard_configs(configs)

        del engine_worker
        torch.cuda.empty_cache()
        MPI.COMM_WORLD.Barrier()
        print(
            f"Memory after cleaning {torch.cuda.memory_allocated()/(1024**3)} GB"
        )

    def generate_tensor_parallel_intervals(self, num_gpus, tp_degree):
        """
        Generate intervals for tensor parallel partitions across available GPUs.
        
        Args:
            num_gpus (int): Number of available GPUs
            tp_degree (int): Tensor parallel degree
        
        Returns:
            list: List of lists containing the partition intervals
        """
        intervals = []
        start = 0

        while start < tp_degree:
            end = min(start + num_gpus, tp_degree)
            interval = list(range(start, end))
            intervals.append(interval)
            start = end

        return intervals

    def run_sharding(self):
        try:
            device_count = torch.cuda.device_count()
            # This is to generate shards by batch
            # Example 1: TP=4, PP=2 on 4-GPU instance
            #   batch 1: PP=0, TP=[0,1,2,3]
            #   batch 2: PP=1, TP=[0,1,2,3]
            # Example 2: TP=8, PP=1 on 4-GPU instance
            #   batch 1: PP=0, TP=[0,1,2,3]
            #   batch 2: PP=0, TP=[4,5,6,7]
            for pp_rank in range(self.pp_degree):
                for tp_interval in self.generate_tensor_parallel_intervals(
                        device_count, self.tp_degree):
                    self.shard_lmi_dist_model(
                        input_dir=self.INPUT_MODEL_DIRECTORY,
                        output_dir=self.OUTPUT_MODEL_DIRECTORY,
                        pp_degree=self.pp_degree,
                        tp_degree=self.tp_degree,
                        chunk_mb=CHUNK_MB,
                        target_pp_rank=pp_rank,
                        target_tp_rank_interval=tp_interval)
            if comms.rank == 0:
                self.save_configs(input_dir=self.INPUT_MODEL_DIRECTORY,
                                  output_dir=self.OUTPUT_MODEL_DIRECTORY)

        except Exception as exc:
            raise OptimizationFatalError(
                f"Encountered an error during sharding: {exc}")


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO,
                        force=True)

    try:
        neo_sharding_service = NeoShardingService()
        neo_sharding_service.run_sharding()
        neo_sharding_service.generate_properties_file()

    except Exception as exc:
        MPI.COMM_WORLD.Barrier()
        write_error_to_file(exc, neo_sharding_service.COMPILATION_ERROR_FILE)
        raise exc
    finally:
        MPI.Finalize()


if __name__ == "__main__":
    main()
