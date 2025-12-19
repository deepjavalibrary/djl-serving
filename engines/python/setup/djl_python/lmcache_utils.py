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
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


def get_numa_node_memory() -> float:
    """
    Get available memory size of a single NUMA node in GB.
    
    Returns:
        float: Available memory size of first NUMA node in GB, or total system memory as fallback
    """
    try:
        result = subprocess.run(['numactl', '-H'],
                                capture_output=True,
                                text=True,
                                timeout=10)

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'node 0 free:' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'free:' and i + 1 < len(parts):
                            free_mb = int(parts[i + 1])
                            free_gb = free_mb / 1024.0
                            logger.info(
                                f"NUMA node 0 available memory: {free_gb:.1f}GB"
                            )
                            return free_gb

    except Exception as e:
        logger.warning(f"NUMA detection failed: {e}")

    # Fallback to total system memory for any failure or non-NUMA systems
    logger.info("Using total system memory instead of NUMA node memory")
    return get_total_cpu_memory()


def get_total_cpu_memory() -> float:
    """
    Get total available CPU memory in GB.
    
    Returns:
        float: Total available CPU memory in GB
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    logger.info(f"Total available CPU memory: {mem_gb:.1f}GB")
                    return mem_gb
    except Exception as e:
        logger.error(f"Failed to get CPU memory: {e}")
        raise


def get_cache_config(properties: Dict[str, str],
                     model_size_gb: float) -> float:
    """
    Calculate cache configuration using appropriate CPU memory based on TP degree.
    
    For TP=1: Uses single NUMA node memory to avoid CUDA allocation limits
    For TP>1: Uses total system memory since it's distributed across processes
    
    Args:
        properties: Serving properties
        model_size_gb: Model size in GB
        
    Returns:
        float: cpu_cache_gb_per_gpu
    """
    tp_degree = int(properties.get("tensor_parallel_degree", 1))

    if tp_degree == 1:
        # For single process (TP=1), use single NUMA node memory to avoid CUDA limits
        total_cpu_memory = get_numa_node_memory()
    else:
        # For multi-process (TP>1), use total system memory since it's distributed
        total_cpu_memory = get_total_cpu_memory()

    total_reserved = max(2 * model_size_gb, 0.2 * total_cpu_memory)
    total_cache_memory = max(0, total_cpu_memory - total_reserved)

    # Divide total cache by TP degree to get cache size per LMCache worker
    cpu_cache_gb_per_gpu = total_cache_memory / tp_degree

    logger.info(f"Cache calculation summary:")
    logger.info(f"  Model size: {model_size_gb:.1f}GB")
    logger.info(f"  Available CPU memory: {total_cpu_memory:.1f}GB")
    logger.info(f"  Total reserved: {total_reserved:.1f}GB")
    logger.info(
        f"  Total available cache memory: {total_cache_memory:.1f}GB ({total_cache_memory/total_cpu_memory:.1%} of available)"
    )
    logger.info(
        f"  Cache per GPU (TP={tp_degree}): {cpu_cache_gb_per_gpu:.1f}GB")

    return cpu_cache_gb_per_gpu


def get_disk_space_info() -> float:
    """
    Get available disk space in GB for LMCache storage.
    
    Returns:
        float: total_disk_gb
    """
    try:
        stat = shutil.disk_usage('/tmp/lmcache')
        total_disk_gb = stat.total / (1024**3)
        logger.info(
            f"Using /tmp/lmcache with {total_disk_gb:.1f}GB total space")
    except Exception as e:
        logger.warning(f"Could not get disk usage for /tmp/lmcache: {e}")
        total_disk_gb = 0

    logger.info(f"Disk space: {total_disk_gb:.1f}GB")
    return total_disk_gb


def calculate_model_size_meta_device(model_id_or_path: str) -> Tuple[int, str]:
    """
    Calculate model size using meta device.
    
    Args:
        model_id_or_path: HuggingFace model ID or local path to model
        
    Returns:
        Tuple[int, str]: (total_parameters, dtype)
    """
    try:
        logger.info(f"Calculating model size for: {model_id_or_path}")

        # Load config from model_id or local path
        config = AutoConfig.from_pretrained(model_id_or_path)

        dtype = getattr(config, 'torch_dtype', 'float32')
        dtype_str = str(dtype).split('.')[-1] if '.' in str(dtype) else str(
            dtype)

        with torch.device("meta"):
            model = AutoModel.from_config(config)

        total_params = sum(p.numel() for p in model.parameters())

        logger.info(
            f"Model size: {total_params:,} parameters, dtype: {dtype_str}")
        return total_params, dtype_str

    except Exception as e:
        logger.error(f"Failed to calculate model size using meta device: {e}")
        raise RuntimeError(
            f"Unable to calculate model size for '{model_id_or_path}'. "
            f"Model size calculation failed with error: {e}. "
            f"Please ensure the model is accessible and has a valid config.json file."
        )


def create_lmcache_config_file(cpu_cache_gb: float, disk_cache_gb: float,
                               model_dir: str) -> str:
    """
    Create LMCache YAML configuration file with calculated sizes.
    
    Args:
        cpu_cache_gb: CPU cache size in GB
        disk_cache_gb: Disk cache size in GB  
        model_dir: Directory to save the config file
        
    Returns:
        str: Path to created config file
    """
    config = {
        'max_local_cpu_size': int(cpu_cache_gb),
        'max_local_disk_size': int(disk_cache_gb),
        'local_disk': 'file:///tmp/lmcache/',
        'extra_config': {
            'use_odirect': True
        }
    }

    config_path = os.path.join(model_dir, "lmcache_auto_config.yaml")

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created LMCache config: {config_path}")
    logger.info(f"Config contents: {config}")

    return config_path


def get_model_size_gb(model_id_or_path: str) -> float:
    """
    Get model size in GB from parameters and dtype.
    
    Args:
        model_id_or_path: HuggingFace model ID or local path
        
    Returns:
        float: Model size in GB
    """
    total_params, dtype_str = calculate_model_size_meta_device(
        model_id_or_path)

    # Calculate bytes per parameter based on dtype
    dtype_bytes = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
        'int4': 0.5
    }

    bytes_per_param = dtype_bytes.get(dtype_str, 4)
    total_bytes = total_params * bytes_per_param
    model_size_gb = total_bytes / (1024**3)

    logger.info(
        f"Model size: {model_size_gb:.1f}GB ({total_params:,} params × {bytes_per_param} bytes)"
    )
    return model_size_gb


def apply_lmcache_auto_config(model_path: str,
                              properties: Dict[str, str]) -> Dict[str, str]:
    """
    Apply LMCache auto-configuration using total CPU memory.
    
    This function:
    1. Checks if lmcache_auto_config=true
    2. Adds kv_transfer_config for LMCache
    3. Calculates model size
    4. Determines cache configuration using total available CPU memory
    5. Creates LMCache config file
    6. Sets LMCACHE_CONFIG_FILE environment variable
    
    Args:
        model_path: Path to model (local directory or HuggingFace model ID)
        properties: Dictionary of serving properties
        
    Returns:
        Dict[str, str]: Updated properties with LMCache configuration
    """
    # Check if auto-config is enabled
    if not properties.get("lmcache_auto_config", "false").lower() == "true":
        return properties

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

        cpu_cache_gb = get_cache_config(properties, model_size_gb)

        disk_cache_gb = get_disk_space_info()

        model_dir = properties.get("model_dir", ".")
        config_file_path = create_lmcache_config_file(cpu_cache_gb,
                                                      disk_cache_gb, model_dir)

        # Set environment variable for LMCache to find the config
        os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath(config_file_path)
        logger.info(
            f"Set LMCACHE_CONFIG_FILE environment variable: {os.environ['LMCACHE_CONFIG_FILE']}"
        )

        logger.info("LMCache auto-configuration completed successfully")
        logger.info(
            f"Final configuration: CPU cache={cpu_cache_gb:.1f}GB, Disk cache={disk_cache_gb:.1f}GB"
        )

    except Exception as e:
        logger.error(f"LMCache auto-configuration failed: {e}")
        logger.warning("Continuing without LMCache auto-configuration")

    return updated_properties
