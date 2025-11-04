# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://aws.amazon.com/apache2.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import time
import json
from typing import Optional

from tensorrt_llm.auto_parallel import infer_cluster_config
from tensorrt_llm.commands.build import parse_arguments, parallel_build
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.plugin import PluginConfig, add_plugin_argument
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig, QuantConfig, CalibConfig, QuantAlgo

from utils import update_kwargs_with_env_vars, load_properties, remove_option_from_properties


def build_engine(
    trtllm_engine_configs: Dict,
    model_id: str,
    output_dir: str,
    tensor_parallel_degree: int,
):

    tik = time.time()
    llm_model = LLM(model=model_id,
                    tensor_parallel_size=tensor_parallel_degree,
                    trust_remote_code=trtllm_engine_configs.get(
                        "trust_remote_code", False),
                    dtype=trtllm_engine_configs.get("dtype", "auto"),
                    revision=trtllm_engine_configs.get("revision", None),
                    **trtllm_engine_configs.llm_kwargs)

    logger.info(f"[LMI] Model Compiled successfully, saving to {output_dir}")
    llm_model.save(output_dir)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


def parse_build_config(properties: dict) -> Dict:
    if "max_rolling_batch_size" in properties:
        properties["max_batch_size"] = properties["max_rolling_batch_size"]
    trtllm_args = []
    for k, v in properties.items():
        trtllm_args.append(f"--{k}")
        trtllm_args.append(f"{v}")
    parser = parse_arguments()
    args, unknown = parser.parse_known_args(args=trtllm_args)
    logger.info(
        f"[LMI] The following args will be passed to the build_config for TRTLLM: {args}"
    )
    logger.info(
        f"[LMI] The following args are not used by TRTLLM build and will be saved for the runtime configuration: {unknown}"
    )

    if hasattr(args, 'gather_generation_logits'):
        logger.warning(
            'Option --gather_generation_logits is deprecated, a build flag is not required anymore. Use --output_generation_logits at runtime instead.'
        )

    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True
    if args.gather_context_logits and args.max_draft_len > 0:
        raise RuntimeError(
            "Gather context logits is not support with draft len > 0. "
            "If want to get the accepted tokens' logits from target model, please just enable gather_generation_logits"
        )

    if hasattr(args, 'paged_kv_cache'):
        logger.warning(
            'Option --paged_kv_cache is deprecated, use --kv_cache_type=paged/disabled instead.'
        )

    plugin_config = PluginConfig.from_arguments(args)
    plugin_config.validate()
    if args.fast_build:
        plugin_config.manage_weights = True

    speculative_decoding_mode = SpeculativeDecodingMode.from_arguments(args)

    if args.build_config is None:
        if args.multiple_profiles == "enable" and args.opt_num_tokens is not None:
            raise RuntimeError(
                "multiple_profiles is enabled, while opt_num_tokens is set. "
                "They are not supposed to be working in the same time for now."
            )
        if args.cluster_key is not None:
            cluster_config = dict(cluster_key=args.cluster_key)
        else:
            cluster_config = infer_cluster_config()

        # This should only be used for debugging.
        # The env var BUILDER_FORCE_NUM_PROFILES should override the number of
        # optimization profiles during TRT build.
        # BUILDER_FORCE_NUM_PROFILES must be less than or equal to the number of
        # optimization profiles set by model's prepare_inputs().
        force_num_profiles_from_env = os.environ.get(
            "BUILDER_FORCE_NUM_PROFILES", None)
        if force_num_profiles_from_env is not None:
            logger.warning(
                f"Overriding # of builder profiles <= {force_num_profiles_from_env}."
            )

        build_config = BuildConfig.from_dict(
            {
                'max_input_len': args.max_input_len,
                'max_seq_len': args.max_seq_len,
                'max_batch_size': args.max_batch_size,
                'max_beam_width': args.max_beam_width,
                'max_num_tokens': args.max_num_tokens,
                'opt_num_tokens': args.opt_num_tokens,
                'max_prompt_embedding_table_size':
                args.max_prompt_embedding_table_size,
                'gather_context_logits': args.gather_context_logits,
                'gather_generation_logits': args.gather_generation_logits,
                'strongly_typed': True,
                'force_num_profiles': force_num_profiles_from_env,
                'weight_sparsity': args.weight_sparsity,
                'profiling_verbosity': args.profiling_verbosity,
                'enable_debug_output': args.enable_debug_output,
                'max_draft_len': args.max_draft_len,
                'speculative_decoding_mode': speculative_decoding_mode,
                'input_timing_cache': args.input_timing_cache,
                'output_timing_cache': '/tmp/model.cache',
                'auto_parallel_config': {
                    'world_size':
                    args.auto_parallel,
                    'gpus_per_node':
                    args.gpus_per_node,
                    'sharded_io_allowlist': [
                        'past_key_value_\\d+',
                        'present_key_value_\\d*',
                    ],
                    'same_buffer_io': {
                        'past_key_value_(\\d+)': 'present_key_value_\\1',
                    },
                    **cluster_config,
                },
                'dry_run': args.dry_run,
                'visualize_network': args.visualize_network,
                'max_encoder_input_len': args.max_encoder_input_len,
                'weight_streaming': args.weight_streaming,
                'monitor_memory': args.monitor_memory,
            },
            plugin_config=plugin_config)

        if hasattr(args, 'kv_cache_type'):
            build_config.update_from_dict(
                {'kv_cache_type': args.kv_cache_type})
    else:
        build_config = BuildConfig.from_json_file(args.build_config,
                                                  plugin_config=plugin_config)
    return build_config.to_dict()


def parse_quant_config(properties: dict) -> Dict:
    quant_config = {}
    if "quant_algo" in properties:
        quant_config["quant_algo"] = QuantAlgo(
            properties.pop("quant_algo").upper())
    if "kv_cache_quant_algo" in properties:
        quant_config["kv_cache_quant_algo"] = QuantAlgo(
            properties.pop("kv_cache_quant_algo").upper())
    if "group_size" in properties:
        quant_config["group_size"] = int(properties.pop("group_size"))
    if "smoothquant_val" in properties:
        quant_config["smoothquant_val"] = float(
            properties.pop("smoothquant_val"))
    if "clamp_val" in properties:
        quant_config["clamp_val"] = json.loads(properties.pop("clamp_val"))
    if "use_meta_recipe" in properties:
        quant_config["use_meta_recipe"] = properties.pop(
            "use_meta_recipe").lower() == "true"
    if "has_zero_point" in properties:
        quant_config["has_zero_point"] = properties.pop(
            "has_zero_point").lower() == "true"
    if "pre_quant_scales" in properties:
        quant_config["pre_quant_scales"] = properties.pop(
            "pre_quant_scales").lower() == "true"
    if "exclude_modules" in properties:
        quant_config["exclude_modules"] = json.loads(
            properties.pop("exclude_modules"))

    return quant_config


def parse_calib_config(properties: dict) -> Dict:
    calib_config = {}
    if "device" in properties:
        calib_config["device"] = properties.pop("device")
    if "calib_dataset" in properties:
        calib_config["calib_dataset"] = properties.pop("calib_dataset")
    if "calib_batches" in properties:
        calib_config["calib_batches"] = int(properties.pop("calib_batches"))
    if "calib_batch_size" in properties:
        calib_config["calib_batch_size"] = int(
            properties.pop("calib_batch_size"))
    if "calib_max_seq_length" in properties:
        calib_config["calib_max_seq_length"] = int(
            properties.pop("calib_max_seq_length"))
    if "random_seed" in properties:
        calib_config["random_seed"] = int(properties.pop("random_seed"))
    if "tokenizer_max_seq_length" in properties:
        calib_config["tokenizer_max_seq_length"] = int(
            properties.pop("tokenizer_max_seq_length"))

    return calib_config


def parse_llm_kwargs(properties: dict) -> dict:
    llm_kwargs = {}
    if "dtype" in properties:
        llm_kwargs["dtype"] = properties.pop("dtype")
    if "revision" in properties:
        llm_kwargs["revision"] = properties.pop("revision")
    if "trust_remote_code" in properties:
        llm_kwargs["trust_remote_code"] = properties.pop("trust_remote_code")
    return llm_kwargs


def generate_trtllm_build_configs(properties: dict) -> Dict:
    quant_config = parse_quant_config(properties)
    calib_config = parse_calib_config(properties)
    llm_kwargs = parse_llm_kwargs(properties)
    build_config = parse_build_config(properties)
    if quant_config:
        llm_kwargs["QuantConfig"] = quant_config

    if calib_config:
        llm_kwargs["CalibConfig"] = calib_config

    if build_config:
        llm_kwargs["BuildConfig"] = build_config

    return llm_kwargs


def sanitize_serving_properties(model_dir: str) -> dict:
    properties = update_kwargs_with_env_vars({})
    properties.update(load_properties(model_dir))
    properties = remove_option_from_properties(properties)
    return properties


def copy_properties_to_compiled_model_dir(source_path: str, dest_path: str):
    with open(os.path.join(source_path, 'serving.properties'),
              'r') as source, open(
                  os.path.join(dest_path, 'serving.properties'), 'w+') as dest:
        for line in source:
            if "option.model_id" in line:
                continue
            dest.write(line)


def main():
    logger.set_level('info')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--properties_dir',
        type=str,
        required=True,
        help='path of the model directory containing properties file')
    parser.add_argument(
        '--trt_llm_model_repo',
        type=str,
        required=True,
        help='local path where trt llm model repo will be created')
    parser.add_argument('--model_path',
                        type=str,
                        required=False,
                        default=None,
                        help='local path to downloaded model')
    parser.add_argument('--tensor_parallel_degree',
                        type=int,
                        required=True,
                        help="tensor parallel degree for compilation")
    parser.add_argument('--pipeline_parallel_degree',
                        type=int,
                        required=True,
                        help="pipeline parallel degree for compilation")

    args = parser.parse_args()
    sanitized_properties = sanitize_serving_properties(args.properties_dir)
    trt_build_configs = generate_trtllm_build_configs(sanitized_properties)
    args.update(trt_build_configs)
    build_engine(
        trt_build_configs,
        args.model_path,
        args.trt_llm_model_repo,
        args.tensor_parallel_degree,
        args.pipeline_parallel_degree,
    )
    copy_properties_to_compiled_model_dir(args.properties_dir,
                                          args.trt_llm_model_repo)


if __name__ == '__main__':
    main()
