#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import re
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Optional

import psutil
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

LMCACHE_DIR = '/tmp/cache'


def get_available_cpu_memory() -> float:
    """
    Get total available CPU memory in GB.
    
    Returns:
        float: Total available CPU memory in GB
    """
    try:
        mem = psutil.virtual_memory()
        mem_gb = mem.available / (1024**3)
        logger.info(f"Total available CPU memory: {mem_gb:.1f}GB")
        return mem_gb
    except Exception as e:
        logger.error(f"Failed to get CPU memory: {e}")
        raise


def calculate_cpu_cache_size(properties: Dict[str, str],
                             model_size_gb: float) -> float:
    """
    Calculate CPU cache size for LMCache based on available memory and TP degree.
    
    Args:
        properties: Serving properties
        model_size_gb: Model size in GB
        
    Returns:
        float: cpu_cache_gb_per_gpu
    """
    tp_degree = int(properties.get("tensor_parallel_degree", 1))

    total_cpu_memory = get_available_cpu_memory()

    # Reserve space: max of 2x model size or 20% of total disk
    total_reserved = max(2 * model_size_gb, 0.2 * total_cpu_memory)
    total_cache_memory = max(0, total_cpu_memory - total_reserved)

    # Divide total cache by TP degree to get cache size per LMCache engine (1 engine per VLLM worker, # of VLLM workers = TP * DP)
    cpu_cache_gb_per_gpu = total_cache_memory / tp_degree

    logger.info(f"CPU cache calculation:")
    logger.info(f"  Model size: {model_size_gb:.1f}GB")
    logger.info(f"  Total CPU memory: {total_cpu_memory:.1f}GB")
    logger.info(
        f"  Reserved CPU memory: {total_reserved:.1f}GB (max of 2x model or 20% of total)"
    )
    logger.info(f"  Total CPU cache: {total_cache_memory:.1f}GB")
    logger.info(
        f"  CPU cache per worker (TP={tp_degree}): {cpu_cache_gb_per_gpu:.1f}GB"
    )

    return cpu_cache_gb_per_gpu


def calculate_disk_cache_size(properties: Dict[str, str],
                             model_size_gb: float) -> float:
    """
    Calculate disk cache size for LMCache based on available disk space and TP degree.
    
    Args:
        properties: Serving properties
        model_size_gb: Model size in GB
        
    Returns:
        float: disk_cache_gb_per_gpu
    """
    tp_degree = int(properties.get("tensor_parallel_degree", 1))

    try:
        stat = shutil.disk_usage('/tmp')
        total_disk_gb = stat.total / (1024**3)
        logger.info(f"Total disk space: {total_disk_gb:.1f}GB")
    except Exception as e:
        logger.warning(f"Could not get disk usage: {e}")
        return 0

    # Reserve space: max of 2x model size or 20% of total disk
    total_reserved = max(2 * model_size_gb, 0.2 * total_disk_gb)
    total_cache_disk = max(0, total_disk_gb - total_reserved)

    # Divide total cache by TP degree to get cache size per LMCache engine (1 engine per VLLM worker, # of VLLM workers = TP * DP)
    disk_cache_gb_per_gpu = total_cache_disk / tp_degree

    logger.info(f"Disk cache calculation:")
    logger.info(f"  Model size: {model_size_gb:.1f}GB")
    logger.info(f"  Total disk: {total_disk_gb:.1f}GB")
    logger.info(
        f"  Reserved disk: {total_reserved:.1f}GB (max of 2x model or 20% of total)"
    )
    logger.info(f"  Total disk cache: {total_cache_disk:.1f}GB")
    logger.info(
        f"  Disk cache per worker (TP={tp_degree}): {disk_cache_gb_per_gpu:.1f}GB"
    )

    return disk_cache_gb_per_gpu


def get_directory_size_gb(path: str) -> float:
    """
    Calculate total size of model files in a directory in GB.
    
    Only counts safetensors OR .bin files to avoid overcounting.
    
    Args:
        path: Path to directory
        
    Returns:
        float: Total size in GB
    """
    safetensors_size = 0
    bin_size = 0

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)

                if filename.endswith('.safetensors'):
                    safetensors_size += file_size
                elif filename.endswith('.bin'):
                    bin_size += file_size

    if safetensors_size > 0:
        total_size = safetensors_size
    elif bin_size > 0:
        total_size = bin_size
    else:
        logger.warning(f"No .safetensors or .bin files found in {path}")
        total_size = 0

    size_gb = total_size / (1024**3)
    logger.info(f"Model file size for {path}: {size_gb:.2f}GB")
    return size_gb


def calculate_model_size_from_hf_api(model_id: str) -> float:
    """
    Calculate model size using HuggingFace Hub API.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')
        
    Returns:
        float: Model size in GB
    """
    try:
        logger.info(
            f"Fetching model info from HuggingFace Hub API: {model_id}")

        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)

        safetensors_size = 0
        bin_size = 0

        if hasattr(info, 'siblings') and info.siblings:
            for sibling in info.siblings:
                filename = sibling.rfilename
                size_bytes = sibling.size or 0

                if filename.endswith('.safetensors'):
                    safetensors_size += size_bytes
                elif filename.endswith('.bin'):
                    bin_size += size_bytes

        if safetensors_size > 0:
            total_size_bytes = safetensors_size
            logger.info(f"Using safetensors files for size calculation")
        elif bin_size > 0:
            total_size_bytes = bin_size
            logger.info(f"Using .bin files for size calculation")
        else:
            raise ValueError("No model files found in model metadata")

        size_gb = total_size_bytes / (1024**3)
        logger.info(f"Model size from HF API: {size_gb:.2f}GB")
        return size_gb

    except Exception as e:
        logger.warning(f"Failed to get model size from HF API: {e}")
        raise


def set_lmcache_env_vars(cpu_cache_gb: float, disk_cache_gb: float) -> None:
    """
    Set LMCache environment variables with calculated sizes.
    
    This configures LMCache directly via environment variables instead of using
    a YAML config file.
    
    Args:
        cpu_cache_gb: CPU cache size in GB
        disk_cache_gb: Disk cache size in GB
    """
    # Core cache size settings
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(int(cpu_cache_gb))
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(int(disk_cache_gb))
    os.environ["LMCACHE_LOCAL_DISK"] = f'file://{LMCACHE_DIR}/'

    # Enable lazy memory allocator for large CPU caches
    os.environ["LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR"] = "true"

    # Set PYTHONHASHSEED for deterministic hashing (required for LMCache)
    os.environ["PYTHONHASHSEED"] = "0"

    # Extra config for O_DIRECT (better disk I/O performance)
    os.environ["LMCACHE_EXTRA_CONFIG"] = '{"use_odirect": true}'

    logger.info(f"Set LMCache environment variables:")
    logger.info(f"  LMCACHE_MAX_LOCAL_CPU_SIZE={int(cpu_cache_gb)}")
    logger.info(f"  LMCACHE_MAX_LOCAL_DISK_SIZE={int(disk_cache_gb)}")
    logger.info(f"  LMCACHE_LOCAL_DISK={LMCACHE_DIR}/")
    logger.info(f"  LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR=true")
    logger.info(f"  PYTHONHASHSEED=0")
    logger.info(f"  LMCACHE_EXTRA_CONFIG={{'use_odirect': true}}")


def get_model_size_gb(model_id_or_path: str) -> float:
    """
    Get model size in GB using the best available method.
    
    Two-option approach:
    1. If local directory: Get directory size
    2. If HuggingFace model ID: Use HF API 
    
    Args:
        model_id_or_path: HuggingFace model ID or local path
        
    Returns:
        float: Model size in GB
    """
    if os.path.isdir(model_id_or_path):
        size_gb = get_directory_size_gb(model_id_or_path)
        if size_gb > 0:
            return size_gb
        else:
            raise RuntimeError(
                f"No model files found in directory '{model_id_or_path}'. "
                f"Please ensure the directory contains model artifacts.")

    try:
        return calculate_model_size_from_hf_api(model_id_or_path)
    except Exception as e:
        logger.error(
            f"HF API method failed for model ID '{model_id_or_path}': {e}")
        raise RuntimeError(
            f"Unable to calculate model size for HuggingFace model ID '{model_id_or_path}'. "
            f"Please ensure the model ID is correct and accessible.")


def apply_lmcache_auto_config(model_path: str,
                              properties: Dict[str, str]) -> Dict[str, str]:
    """
    Apply LMCache auto-configuration using total CPU memory.
    
    This function:
    1. Checks if lmcache_auto_config=true
    2. Adds kv_transfer_config for LMCache
    3. Calculates model size
    4. Determines cache configuration using total available CPU memory
    5. Sets LMCache environment variables directly
    
    Args:
        model_path: Path to model (local directory or HuggingFace model ID)
        properties: Dictionary of serving properties
        
    Returns:
        Dict[str, str]: Updated properties with LMCache configuration
    """
    # Check if auto-config is enabled
    if not properties.get("lmcache_auto_config", "false").lower() == "true":
        return properties

    # Fail fast if expert parallelism is enabled
    # EP changes worker count calculation (TP × DP), which auto-config doesn't currently support
    if properties.get("enable_expert_parallel", "false").lower() == "true":
        raise RuntimeError(
            "LMCache auto-configuration does not currently support expert parallelism (option.enable_expert_parallel=true). "
            "Either disable expert parallelism or use manual LMCache configuration with option.lmcache_config_file"
        )

    # Fail fast if lmcache_config_file property is set (manual config file)
    if "lmcache_config_file" in properties:
        raise RuntimeError(
            f"LMCache auto-configuration cannot proceed: option.lmcache_config_file is set to '{properties['lmcache_config_file']}'. "
            f"Auto-configuration is incompatible with manual LMCache configuration files. "
            f"Either remove option.lmcache_config_file or disable auto-configuration "
            f"by setting option.lmcache_auto_config=false")

    # Warn if LMCACHE_* env vars are already set (they will be overwritten)
    lmcache_env_vars = {
        k: v
        for k, v in os.environ.items() if k.startswith("LMCACHE_")
    }
    if lmcache_env_vars:
        env_var_list = ", ".join(
            f"{k}={v}" for k, v in list(lmcache_env_vars.items())[:3])
        logger.warning(
            f"LMCache auto-configuration detected existing LMCACHE environment variables: {env_var_list}. "
            f"Applicable LMCache configuration variables will be overwritten by auto-configuration.")

    updated_properties = properties.copy()

    try:
        # Add kv_transfer_config for LMCache integration
        updated_properties["kv_transfer_config"] = json.dumps({
            "kv_connector":
            "LMCacheConnectorV1",
            "kv_role":
            "kv_both"
        })
        logger.info("Added kv_transfer_config for LMCache")

        model_size_gb = get_model_size_gb(model_path)

        cpu_cache_gb = calculate_cpu_cache_size(properties, model_size_gb)

        disk_cache_gb = calculate_disk_cache_size(properties, model_size_gb)

        # Set LMCache configuration via environment variables
        set_lmcache_env_vars(cpu_cache_gb, disk_cache_gb)

        logger.info("LMCache auto-configuration completed successfully")
        logger.info(
            f"Final configuration: CPU cache={cpu_cache_gb:.1f}GB, Disk cache={disk_cache_gb:.1f}GB"
        )

    except Exception as e:
        logger.error(f"LMCache auto-configuration failed: {e}")
        raise RuntimeError(
            f"LMCache auto-configuration failed: {e}. "
            f"Disable auto-configuration by setting option.lmcache_auto_config=false "
            f"and provide a manual LMCache configuration.")

    return updated_properties
