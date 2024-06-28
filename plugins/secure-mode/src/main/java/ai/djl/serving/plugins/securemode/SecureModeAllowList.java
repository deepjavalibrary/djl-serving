package ai.djl.serving.plugins.securemode;

import java.util.Set;

/** A class for properties that are allowlisted in Secure Mode. */
public class SecureModeAllowList {
    public static final Set<String> PROPERTIES_ALLOWLIST =
            Set.of(
                    "engine",
                    "job_queue_size",
                    "option.entryPoint",
                    "option.model_id",
                    "option.batch_size",
                    "option.tensor_parallel_degree",
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
                    "option.max_cpu_loras");
}
