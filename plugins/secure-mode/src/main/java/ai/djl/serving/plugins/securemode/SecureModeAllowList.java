/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.plugins.securemode;

import java.util.Set;

/** Properties that are explicitly allowlisted in Secure Mode. */
interface SecureModeAllowList {

    public static final Set<String> PROPERTIES_ALLOWLIST =
            Set.of(
                    "engine",
                    "job_queue_size",
                    "max_idle_time",
                    "batch_size",
                    "max_batch_delay",
                    "max_dynamic_batch_size",
                    "minWorkers",
                    "maxWorkers",
                    "load_on_devices",
                    "option.mpi_mode",
                    "option.entryPoint",
                    "option.task",
                    "option.model_id",
                    "option.batch_size",
                    "option.tensor_parallel_degree",
                    "option.pipeline_parallel_degree",
                    "option.rolling_batch",
                    "option.dtype",
                    "option.trust_remote_code",
                    "option.revision",
                    "option.max_rolling_batch_size",
                    "option.parallel_loading",
                    "option.model_loading_timeout",
                    "option.output_formatter",
                    "option.n_positions",
                    "option.load_in_8bit",
                    "option.unroll",
                    "option.neuron_optimize_level",
                    "option.context_length_estimate",
                    "option.low_cpu_mem_usage",
                    "option.load_split_model",
                    "option.compiled_graph_path",
                    "option.group_query_attention",
                    "option.enable_mixed_precision_accumulation",
                    "option.enable_saturate_infinity",
                    "option.speculative_draft_model",
                    "option.speculative_length",
                    "option.draft_model_compiled_path",
                    "option.quantize",
                    "option.max_rolling_batch_prefill_tokens",
                    "option.max_model_len",
                    "option.load_format",
                    "option.enforce_eager",
                    "option.gpu_memory_utilization",
                    "option.draft_model_tp_size",
                    "option.record_acceptance_rate",
                    "option.enable_lora",
                    "option.max_loras",
                    "option.max_lora_rank",
                    "option.lora_extra_vocab_size",
                    "option.max_cpu_loras",
                    "option.max_input_len",
                    "option.max_output_len",
                    "option.max_num_tokens",
                    "option.use_custom_all_reduce",
                    "option.tokens_per_block",
                    "option.batch_scheduler_policy",
                    "option.kv_cache_free_gpu_mem_fraction",
                    "option.max_num_sequences",
                    "option.enable_trt_overlap",
                    "option.enable_kv_cache_reuse",
                    "option.baichuan_model_version",
                    "option.chatglm_model_version",
                    "option.gpt_model_version",
                    "option.multi_block_mode",
                    "option.use_fused_mlp",
                    "option.rotary_base",
                    "option.rotary_dim",
                    "option.rotary_scaling_type",
                    "option.rotary_scaling_factor",
                    "option.logits_dtype",
                    "option.trtllm_checkpoint_path",
                    "option.load_by_shard",
                    "option.smoothquant_alpha",
                    "option.smoothquant_per_token",
                    "option.smoothquant_per_channel",
                    "option.multi_query_mode",
                    "option.awq_format",
                    "option.awq_calib_size",
                    "option.q_format",
                    "option.calib_size",
                    "option.calib_batch_size",
                    "option.use_fp8_context_fmha",
                    "option.enable_chunked_prefill",
                    "option.cpu_offload_gb_per_gpu",
                    "option.enable_prefix_caching",
                    "option.disable_sliding_window",
                    "option.enable_streaming",
                    "option.tgi_compat",
                    "option.pythonExecutable");

    public static final Set<String> PYTHON_EXECUTABLE_ALLOWLIST =
            Set.of("/opt/djl/vllm_venv/bin/python", "/usr/bin/python3");
}
