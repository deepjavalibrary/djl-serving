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
import logging
from typing import List, Dict, Any
import subprocess as sp

from djl_python.properties_manager.properties import RollingBatchEnum

logger = logging.getLogger(__name__)

BILLION = 1_000_000_000.0
MAX_ROLLING_BATCH = 32  # Current best throughput and latency balance batch size
MEMORY_PER_CORE = 16.0  # Currently there is only one config w/ 16 GB per core


class NeuronSmartDefaultUtils:
    """
    This class provides the Neuron smart default configurations for the model.
    Additionally, it mirrors the Java implementation in the same package. Tests are mirrored
    by ahead of time compilation using Java and loading using this python util.
    """

    def __init__(self,
                 available_cores: int = 0,
                 model_size_in_gb: float = 0,
                 sequence_size_in_gb: float = 0) -> None:
        """
        Initializes the NeuronSmartDefaultUtils class with the given available cores, model size in GB, and sequence size in GB.

        If available_cores is not set, it will be set to the number of available cores on the machine.

        :param available_cores: The number of available cores on the machine
        :param model_size_in_gb: The size of the model in GB
        :param sequence_size_in_gb: The size of the sequence in GB
        """
        if available_cores:
            self.available_cores = available_cores
        else:
            self.available_cores = self.get_available_cores()
        self.model_size_in_gb = model_size_in_gb
        self.sequence_size_in_gb = sequence_size_in_gb

    def apply_smart_defaults(self,
                             properties: Dict[str, Any],
                             model_config: Dict[str, Any],
                             is_partition: bool = False) -> None:
        """
        Applies smart defaults for Neuron models.

        If not already set, this method sets the following properties:
        - n_positions: The default n_positions for the model.
        - tensor_parallel_degree: A heuristic based on available memory.
        - max_rolling_batch_size: A heuristic based on available memory.
        - on_device_embedding: From neuron 2.20, saving and loading pre-sharded weights is
                               only available for on_device_embedding.

        :param properties: The properties to update
        :param model_config: The model configuration to use
        :param is_partition: Indicates whether we are saving pre-sharded checkpoints or not.
                             We set some smart defaults for it.
        """

        if "rolling_batch" not in properties:
            properties["rolling_batch"] = RollingBatchEnum.auto.value

        if "n_positions" not in properties:
            if self.get_model_parameters(
                    model_config) <= 0 or self.available_cores == 0:
                # If n positions cannot be determined, skip smart defaults
                return
            properties["n_positions"] = min(
                model_config.get("max_position_embeddings", 4096), 4096)
            logger.info(
                f"[Smart Default] N_POSITIONS: {properties['n_positions']}.")

        # if neuron exists in config.json, then we could safely assume, it is precomplied model.
        if "neuron" in model_config.keys() or is_partition:
            if "on_device_embedding" not in properties:
                properties["on_device_embedding"] = True

        try:
            self.set_internal_settings(properties, model_config)
        except Exception as e:
            logger.debug(f"Failed to set internal settings: {e}")
            return

        self.set_heuristic_neuron_tp_degree(properties)
        self.set_heuristic_neuron_max_rolling_batch(properties)

    @staticmethod
    def get_available_cores():
        """
        Get the number of available Neuron cores.

        Uses the `neuron-ls --json-output` command to get the list of available Neuron cores.
        If the command fails for any reason, returns 0.

        :return: The number of available Neuron cores
        """
        command = "neuron-ls --json-output"
        try:
            output = sp.check_output(command, shell=True).decode("utf-8")
            return len(json.loads(output))
        except Exception as e:
            logger.debug(f"Failed to get available cores: {e}")
        return 0

    def get_model_parameters(self, model_config):
        """
        Compute the number of parameters in a model.

        Currently only supports LLaMA-like models. For other models, returns 0.

        :param model_config: The model configuration dictionary
        :return: The number of parameters in the model
        """
        if model_config.get("model_type") == "llama" or model_config.get(
                "model_type") == "mistral":
            return self.get_llama_like_parameters(model_config)
        return 0

    @staticmethod
    def get_llama_like_parameters(model_config: Dict[str, Any]) -> int:
        """
        Compute the number of parameters in a LLaMA-like model.

        The parameter count is computed as follows:

        - embedding size: hidden size * vocab size
        - qkv size: head dim * hidden size * num key value heads * 3
        - o proj size: hidden size * hidden size
        - gate proj size: hidden size * intermediate size * 3
        - total size: embedding size + num hidden layers * (qkv size + o proj size + gate proj size + 2 * hidden size) + hidden size + embedding size

        :param model_config: The model configuration dictionary
        :return: The number of parameters in the LLaMA-like model
        """
        try:
            head_dim = model_config["num_attention_heads"] * model_config[
                "num_key_value_heads"]
            embedding_size = model_config["hidden_size"] * model_config[
                "vocab_size"]
            qkv_size = head_dim * model_config["hidden_size"] * model_config[
                "num_key_value_heads"] * 3
            o_proj_size = model_config["hidden_size"] * model_config[
                "hidden_size"]
            gate_proj_size = model_config["hidden_size"] * model_config[
                "intermediate_size"] * 3
            return embedding_size + model_config["num_hidden_layers"] * (
                qkv_size + o_proj_size + gate_proj_size +
                model_config["hidden_size"] + model_config["hidden_size"]
            ) + model_config["hidden_size"] + embedding_size
        except Exception as e:
            logger.debug(f"Failed to get llama-like parameters: {e}")
            return 0

    @staticmethod
    def get_single_sequence_size(sequence_length: int, weight_bytes: int,
                                 model_config: Dict[str, Any]) -> int:
        """
        Compute the memory required to store a single sequence of tokens.

        The memory required is computed as the product of the sequence length,
        hidden size, number of hidden layers, and weight size in bytes.

        :param sequence_length: The length of the sequence in tokens
        :param weight_bytes: The size of a single model weight in bytes
        :param model_config: The model configuration dictionary
        :return: The memory required to store a single sequence of tokens in bytes
        """
        try:
            return sequence_length * model_config["hidden_size"] * model_config[
                "num_hidden_layers"] * weight_bytes
        except Exception as e:
            logger.debug(f"Failed to get single sequence size: {e}")
            return 0

    def set_internal_settings(self, properties: Dict[str, Any],
                              model_config: Dict[str, Any]) -> None:
        """
        Set the internal settings for this smart default.

        Internal settings include the model size in GB and the sequence size in GB.
        These are computed based on the model configuration and the presence of
        quantization.

        :param properties: The properties dictionary
        :param model_config: The model configuration dictionary
        :return: None
        """
        n_positions = int(properties.get("n_positions", 0))
        param_bytes = 1 if "option.quantize" in properties else 2
        self.model_size_in_gb = (
            param_bytes * self.get_model_parameters(model_config)) / BILLION
        self.sequence_size_in_gb = (self.get_single_sequence_size(
            n_positions, param_bytes, model_config) * 0.95 /
                                    (1024.0 * 1024.0 * 1024.0))

        if self.model_size_in_gb == 0 or self.sequence_size_in_gb == 0 or n_positions == 0:
            raise Exception(
                f"Failed to compute model size or sequence size or n_positions: {n_positions},"
                f"model_size_in_gb: {self.model_size_in_gb}, sequence_size_in_gb: {self.sequence_size_in_gb}"
            )

    def get_adjusted_model_size_in_gb(self, tp_degree: int) -> float:
        return self.model_size_in_gb * (1.0 + ((tp_degree * 2 - 2) / 100.0))

    def set_heuristic_neuron_tp_degree(self, properties: Dict[str,
                                                              Any]) -> None:
        """
        Sets a heuristic value for tensor parallel degree if not already set in model properties.

        There are two scenarios where the tensor parallel degree is set:
        - If the tensor parallel degree is not set in the properties and the max_rolling_batch_size is not set,
          then the tensor parallel degree is set based on maximizing instance concurrency with variable rolling batch size.
          The tensor parallel degree is set to the minimum core configuration that supports the maximum instance concurrency.
        - If the tensor parallel degree is not set in the properties and the max_rolling_batch_size is set,
          then the tensor parallel degree is set by minimizing TP degree that supports fixed batch size.
          The tensor parallel degree is set to the minimum core configuration that supports the maximum instance concurrency given the fixed batch size.

        :param properties: The properties dictionary
        :return: None
        """
        tp_degree = self.available_cores
        total_memory = tp_degree * MEMORY_PER_CORE

        if "tensor_parallel_degree" in properties and properties[
                "tensor_parallel_degree"] == "max":
            properties["tensor_parallel_degree"] = self.available_cores
            logger.info(
                f"[Smart Default] TENSOR_PARALLEL_DEGREE: {properties['tensor_parallel_degree']}."
            )
            return

        core_configs = self.available_core_configs()

        if "tensor_parallel_degree" not in properties and "max_rolling_batch_size" not in properties:
            # Set tensor parallel degree based on maximizing instance concurrency with variable rolling batch size
            total_instance_concurrency = self.get_max_concurrency(
                total_memory, tp_degree)
            for core_config in core_configs:
                max_memory = core_config * MEMORY_PER_CORE
                max_concurrency = (
                    self.get_max_concurrency(max_memory, core_config) *
                    (self.available_cores // core_config))
                if max_concurrency >= total_instance_concurrency and core_config <= tp_degree:
                    tp_degree = core_config
                    total_instance_concurrency = max_concurrency
            properties["tensor_parallel_degree"] = tp_degree
            logger.info(
                f"[Smart Default] TENSOR_PARALLEL_DEGREE: {properties['tensor_parallel_degree']}."
            )
        elif "tensor_parallel_degree" not in properties:
            # Set tensor parallel degree by minimizing TP degree that supports fixed batch size
            batch_size = int(properties["max_rolling_batch_size"])
            total_instance_concurrency = self.get_max_concurrency_with_batch(
                total_memory, tp_degree, batch_size)
            for core_config in core_configs:
                max_memory = core_config * MEMORY_PER_CORE
                max_concurrency = (self.get_max_concurrency_with_batch(
                    max_memory, core_config, batch_size) *
                                   (self.available_cores // core_config))
                if max_concurrency >= total_instance_concurrency and core_config <= tp_degree:
                    tp_degree = core_config
                    total_instance_concurrency = max_concurrency
            properties["tensor_parallel_degree"] = tp_degree
            logger.info(
                f"[Smart Default] TENSOR_PARALLEL_DEGREE: {properties['tensor_parallel_degree']}."
            )

    @staticmethod
    def get_max_power_of_2(n: int) -> int:
        """
        Finds the largest power of 2 less than or equal to n.

        :param n: the input number
        :return: the largest power of 2 less than or equal to n
        """
        n = min(n, MAX_ROLLING_BATCH)
        if n != 0 and (n & (n - 1)) == 0:
            return n
        max_power_of_2 = 1
        while max_power_of_2 < n:
            max_power_of_2 *= 2
        return max_power_of_2 // 2

    def get_max_concurrency(self, total_memory: float, tp_degree: int) -> int:
        """
        Calculates the maximum number of concurrent requests that can be served by a model given the
        total memory available for the model and the sequence size.

        The maximum number of concurrent requests is calculated as the largest power of 2 less
        than or equal to the total memory divided by the sequence size.

        :param total_memory: the total memory available for the model
        :param tp_degree: the tensor parallel degree
        :return: the maximum number of concurrent requests
        """
        max_concurrency = int(
            (total_memory - self.get_adjusted_model_size_in_gb(tp_degree)) /
            self.sequence_size_in_gb)
        max_concurrency = self.get_max_power_of_2(max_concurrency)
        return min(max_concurrency, MAX_ROLLING_BATCH)

    def get_max_concurrency_with_batch(self, total_memory: float,
                                       tp_degree: int, batch_size: int) -> int:
        """
        Calculates the maximum number of concurrent requests that can be served by a model given the
        total memory available for the model and the sequence size, and the desired batch size.

        :param total_memory: the total memory available for the model
        :param tp_degree: the tensor parallel degree
        :param batch_size: the desired batch size
        :return: the maximum number of concurrent requests
        """
        max_concurrency = int(
            (total_memory - self.get_adjusted_model_size_in_gb(tp_degree)) /
            self.sequence_size_in_gb)
        max_concurrency = self.get_max_power_of_2(max_concurrency)
        max_concurrency = min(max_concurrency, batch_size)
        return max_concurrency if max_concurrency == batch_size else 0

    def available_core_configs(self) -> List[int]:
        """
        Builds the available core configurations for a given number of cores.

        The available core configurations are those that are less than or equal to the total
        number of cores. This method returns a list of available core configurations for the given
        number of cores.

        :return: the list of available core configurations
        """
        core_configs = self.build_core_configs(self.available_cores)
        cores_per_model = int(1.1 * self.model_size_in_gb / MEMORY_PER_CORE)
        return [config for config in core_configs if cores_per_model <= config]

    @staticmethod
    def build_core_configs(n_cores: int) -> List[int]:
        """
        Builds the available core configurations for a given number of cores.

        The available core configurations are those that are less than or equal to the total
        number of cores. This method returns a list of available core configurations for the given
        number of cores.

        :param n_cores: the number of cores to build the configurations for
        :return: the list of available core configurations
        """
        core_configs = [1, 2, 8]
        if n_cores > 8:
            core_configs.append(n_cores)
        return core_configs

    def set_heuristic_neuron_max_rolling_batch(
            self, properties: Dict[str, Any]) -> None:
        """
        Sets the max rolling batch size based on the TP degree and the model memory size.

        If the max rolling batch size is not set, this method sets it to the maximum number of
        concurrent requests that can be served by a model given the total memory available for the
        model and the sequence size.

        :param properties: The properties dictionary
        :return: None
        """
        tp_degree = int(
            properties.get("tensor_parallel_degree", self.available_cores))
        if "max_rolling_batch_size" not in properties:
            max_rolling_batch_size = self.get_max_concurrency(
                tp_degree * MEMORY_PER_CORE, tp_degree)
            if max_rolling_batch_size > 0:
                properties["max_rolling_batch_size"] = max_rolling_batch_size
                logger.info(
                    f"[Smart Default] MAX_ROLLING_BATCH_SIZE: {properties['max_rolling_batch_size']}."
                )
