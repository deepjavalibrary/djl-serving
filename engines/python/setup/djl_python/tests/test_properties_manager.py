import os
import json
import unittest
from unittest import mock

from vllm import EngineArgs

from djl_python.properties_manager.properties import Properties
from djl_python.properties_manager.tnx_properties import (
    TransformerNeuronXProperties, TnXGenerationStrategy, TnXModelSchema,
    TnXMemoryLayout, TnXDtypeName, TnXModelLoaders)
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.properties_manager.sd_inf2_properties import StableDiffusionNeuronXProperties
from djl_python.tests.utils import parameterized, parameters

import torch

model_min_properties = {"model_id": "model_id", "model_dir": "model_dir"}

min_common_properties = {
    "model_id": "model_id",
    "model_dir": "model_dir",
    "rolling_batch": "auto",
    "tensor_parallel_degree": "4",
}

common_properties = {
    "model_id": "model_id",
    "model_dir": "model_dir",
    "rolling_batch": "disable",
    "tensor_parallel_degree": "4",
    'batch_size': "4",
    'max_rolling_batch_size': '2',
    'enable_streaming': 'False',
    'dtype': 'fp16',
    'revision': 'shdghdfgdfg',
    'trust_remote_code': 'true',
    # spec_dec
    "draft_model_id": "draft_model_id",
    "spec_length": "0"
}


@parameterized
class TestConfigManager(unittest.TestCase):

    def test_common_configs(self):
        configs = Properties(**min_common_properties)
        self.assertEqual(min_common_properties['model_id'],
                         configs.model_id_or_path)
        self.assertEqual(min_common_properties['rolling_batch'],
                         configs.rolling_batch)
        self.assertEqual(int(min_common_properties['tensor_parallel_degree']),
                         configs.tensor_parallel_degree)

        self.assertEqual(configs.batch_size, 1)
        self.assertEqual(configs.max_rolling_batch_size, 32)
        self.assertEqual(configs.enable_streaming.value, 'false')

        self.assertFalse(configs.trust_remote_code)
        self.assertIsNone(configs.dtype)
        self.assertIsNone(configs.revision)

    def test_all_common_configs(self):
        configs = Properties(**common_properties)
        self.assertEqual(configs.batch_size, 4)
        self.assertEqual(configs.tensor_parallel_degree, 4)
        self.assertEqual(common_properties['model_id'],
                         configs.model_id_or_path)
        self.assertEqual(common_properties['rolling_batch'],
                         configs.rolling_batch)
        self.assertEqual(int(common_properties['tensor_parallel_degree']),
                         configs.tensor_parallel_degree)

        self.assertEqual(int(common_properties['batch_size']),
                         configs.batch_size)
        self.assertEqual(int(common_properties['max_rolling_batch_size']),
                         configs.max_rolling_batch_size)
        self.assertEqual(configs.enable_streaming.value, 'false')

        self.assertTrue(configs.trust_remote_code)
        self.assertEqual(configs.dtype, common_properties['dtype'])
        self.assertEqual(configs.revision, common_properties['revision'])

        self.assertEqual(common_properties['draft_model_id'],
                         configs.draft_model_id)
        self.assertEqual(configs.spec_length, 0)

    def test_common_configs_error_case(self):
        other_properties = min_common_properties
        other_properties["rolling_batch"] = "disable"
        other_properties["enable_streaming"] = "true"
        other_properties["batch_size"] = '2'
        with self.assertRaises(ValueError):
            Properties(**other_properties)

    def test_tnx_configs(self):
        properties = {
            "n_positions": "256",
            "load_split_model": "true",
            "quantize": "static_int8",
            "compiled_graph_path": "s3://test/bucket/folder"
        }
        tnx_configs = TransformerNeuronXProperties(**common_properties,
                                                   **properties)
        self.assertFalse(tnx_configs.low_cpu_mem_usage)
        self.assertTrue(tnx_configs.load_split_model)
        self.assertEqual(int(properties['n_positions']),
                         tnx_configs.n_positions)
        self.assertEqual(tnx_configs.tensor_parallel_degree,
                         int(min_common_properties['tensor_parallel_degree']))
        self.assertEqual(tnx_configs.quantize.value, properties['quantize'])
        self.assertTrue(tnx_configs.load_in_8bit)
        self.assertEqual(tnx_configs.batch_size, 4)
        self.assertEqual(tnx_configs.max_rolling_batch_size, 2)
        self.assertEqual(tnx_configs.enable_streaming.value, 'false')
        self.assertEqual(tnx_configs.compiled_graph_path,
                         str(properties['compiled_graph_path']))

    def test_tnx_all_configs(self):
        properties = {
            "n_positions": "2048",
            "load_split_model": "true",
            "load_in_8bit": "true",
            "compiled_graph_path": "s3://test/bucket/folder",
            "low_cpu_mem_usage": "true",
            'context_length_estimate': '256, 512, 1024',
            "task": "feature-extraction",
            "save_mp_checkpoint_path": "/path/to/checkpoint",
            "neuron_optimize_level": 3,
            "enable_mixed_precision_accumulation": "true",
            "group_query_attention": "shard-over-heads",
            "shard_over_sequence": "true",
            "fuse_qkv": "true",
            "fuse_mlp": "true",
            "fused_rmsnorm_qkv": "true",
            "qkv_tiling": "true",
            "weight_tiling": "true",
            "enable_saturate_infinity": "true",
            "rolling_batch_strategy": "continuous_batching",
            "collectives_layout": "HSB",
            "partition_schema": "safetensors",
            "attention_layout": "HSB",
            "cache_layout": "SBH",
            "all_reduce_dtype": "float32",
            "cast_logits_dtype": "float32",
            "on_device_embedding": "true",
            "on_device_generation": "./sample.json",
            "draft_model_compiled_path": "s3://test/bucket/folder",
            "speculative_draft_model": "draft_model_id",
            "speculative_length": 4,
            "draft_model_tp_size": 8,
            "neuron_quant": "true",
            "sequence_parallel": "false",
            "multi_node": "false",
            "neuron_cc_pipeline_factor": 2,
            "compilation_worker_count": 2
        }

        generation_config = {"top_k": 25}

        with open("sample.json", "w") as fp:
            json.dump(generation_config, fp)

        tnx_configs = TransformerNeuronXProperties(**common_properties,
                                                   **properties)
        self.assertEqual(tnx_configs.n_positions, 2048)
        self.assertEqual(tnx_configs.compiled_graph_path,
                         properties['compiled_graph_path'])

        self.assertTrue(tnx_configs.load_split_model)
        self.assertTrue(tnx_configs.load_in_8bit)
        self.assertTrue(tnx_configs.low_cpu_mem_usage)

        self.assertListEqual(tnx_configs.context_length_estimate,
                             [256, 512, 1024])

        self.assertEqual(tnx_configs.task, properties['task'])
        self.assertEqual(tnx_configs.save_mp_checkpoint_path,
                         properties['save_mp_checkpoint_path'])
        neuron_cc = os.environ["NEURON_CC_FLAGS"]
        self.assertTrue("-O3" in neuron_cc)
        self.assertTrue("--enable-mixed-precision-accumulation" in neuron_cc)
        self.assertTrue("--enable-saturate-infinity" in neuron_cc)
        self.assertEqual(tnx_configs.group_query_attention,
                         properties['group_query_attention'])
        self.assertTrue(tnx_configs.shard_over_sequence)
        self.assertTrue(tnx_configs.fuse_qkv)
        self.assertTrue(tnx_configs.fuse_mlp)
        self.assertTrue(tnx_configs.fused_rmsnorm_qkv)
        self.assertTrue(tnx_configs.qkv_tiling)
        self.assertTrue(tnx_configs.weight_tiling)
        self.assertEqual(tnx_configs.rolling_batch_strategy,
                         TnXGenerationStrategy.continuous_batching)
        self.assertEqual(tnx_configs.collectives_layout,
                         TnXMemoryLayout.LAYOUT_HSB)
        self.assertEqual(tnx_configs.partition_schema,
                         TnXModelSchema.safetensors)
        self.assertEqual(tnx_configs.draft_model_compiled_path,
                         properties['draft_model_compiled_path'])
        self.assertEqual(tnx_configs.attention_layout,
                         TnXMemoryLayout.LAYOUT_HSB)
        self.assertEqual(tnx_configs.cache_layout, TnXMemoryLayout.LAYOUT_SBH)
        self.assertEqual(tnx_configs.all_reduce_dtype, TnXDtypeName.float32)
        self.assertEqual(tnx_configs.cast_logits_dtype, TnXDtypeName.float32)
        self.assertEqual(tnx_configs.model_loader, TnXModelLoaders.tnx)
        self.assertTrue(tnx_configs.on_device_embedding)
        self.assertDictEqual(tnx_configs.on_device_generation,
                             generation_config)
        self.assertTrue(tnx_configs.neuron_quant)
        self.assertFalse(tnx_configs.sequence_parallel)
        self.assertFalse(tnx_configs.multi_node)
        self.assertEqual(tnx_configs.neuron_cc_pipeline_factor, 2)
        self.assertEqual(tnx_configs.compilation_worker_count, 2)
        self.assertEqual(tnx_configs.speculative_draft_model,
                         properties['speculative_draft_model'])
        self.assertEqual(tnx_configs.speculative_length,
                         properties['speculative_length'])
        self.assertEqual(tnx_configs.draft_model_tp_size,
                         properties['draft_model_tp_size'])

        # tests context length estimate as integer
        def test_tnx_cle_int(context_length_estimate):
            properties['context_length_estimate'] = context_length_estimate
            configs = TransformerNeuronXProperties(**common_properties,
                                                   **properties)
            self.assertEqual(configs.context_length_estimate, [256])
            del properties['context_length_estimate']

        test_tnx_cle_int('256')
        os.remove("sample.json")

    @parameters([{
        "is_env":
        True,
        "NEURON_ON_DEVICE_EMBEDDING":
        "true",
        "NEURON_ON_DEV_GENERATION":
        "true",
        "NEURON_SHARD_OVER_SEQUENCE":
        "true",
        "NEURON_QUANT":
        "true",
        "NEURON_SEQUENCE_PARALLEL":
        "false",
        "NEURON_MULTI_NODE":
        "false",
        "NEURON_COMPILATION_WORKER_COUNT":
        "2",
        "NEURON_CC_PIPELINE_FACTOR":
        "2",
        "NEURON_CONTEXT_LENGTH_ESTIMATE":
        "[1024, 2048, 4096, 8192, 16384]"
    }])
    def test_neuron_env_configs(self, params):
        is_env = params.pop("is_env", False)
        if is_env:
            properties = min_common_properties
            for param in params:
                os.environ[param] = params[param]
        else:
            properties = {**min_common_properties, **params}

        tnx_configs = TransformerNeuronXProperties(**properties)
        self.assertTrue(tnx_configs.neuron_quant)
        self.assertTrue(tnx_configs.on_device_embedding)
        self.assertTrue(tnx_configs.on_device_generation)
        self.assertTrue(tnx_configs.shard_over_sequence)
        self.assertFalse(tnx_configs.multi_node)
        self.assertFalse(tnx_configs.sequence_parallel)
        self.assertEqual(tnx_configs.compilation_worker_count, 2)
        self.assertEqual(tnx_configs.neuron_cc_pipeline_factor, 2)
        self.assertEqual(tnx_configs.context_length_estimate,
                         [1024, 2048, 4096, 8192, 16384])

        if is_env:
            for param in params:
                del os.environ[param]

    @parameters([{
        "compiled_graph_path": "https://random.url.address/"
    }, {
        "compiled_graph_path": "not_a_directory"
    }, {
        "context_length_estimate": "invalid"
    }, {
        'group_query_attention': "invalid"
    }, {
        'rolling_batch': 'auto'
    }, {
        'model_loader': 'tnx',
        'partition_schema': 'optimum',
        'load_split_model': 'true'
    }, {
        'model_loader': 'optimum'
    }])
    def test_tnx_configs_error_case(self, params):
        # To remove the duplicate properties
        properties = {**common_properties, **params}
        with self.assertRaises(ValueError):
            TransformerNeuronXProperties(**properties)

    @parameters([{
        "rolling_batch": "auto",
    }])
    def test_trt_llm_configs(self, params):
        properties = {**model_min_properties, **params}
        trt_configs = TensorRtLlmProperties(**properties)
        self.assertEqual(trt_configs.model_id_or_path, properties['model_id'])
        self.assertEqual(trt_configs.rolling_batch.value,
                         properties['rolling_batch'])

    def test_hf_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "low_cpu_mem_usage": "true",
            "disable_flash_attn": "false",
            "mpi_mode": "true",
            "rolling_batch": "disable"
        }

        hf_configs = HuggingFaceProperties(**properties)
        self.assertIsNone(hf_configs.load_in_8bit)
        self.assertIsNone(hf_configs.device)
        self.assertTrue(hf_configs.low_cpu_mem_usage)
        self.assertFalse(hf_configs.disable_flash_attn)
        self.assertIsNone(hf_configs.device_map)
        self.assertTrue(hf_configs.mpi_mode)
        self.assertDictEqual(hf_configs.kwargs, {
            'trust_remote_code': False,
            "low_cpu_mem_usage": True,
        })

    def test_hf_all_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "tensor_parallel_degree": "4",
            "cluster_size": "2",
            "load_in_4bit": "false",
            "load_in_8bit": "true",
            "low_cpu_mem_usage": "true",
            "disable_flash_attn": "false",
            "mpi_mode": "true",
            "device_map": "cpu",
            "quantize": "bitsandbytes8",
            "output_formatter": "jsonlines",
            "waiting_steps": '12',
            "trust_remote_code": "true",
            "rolling_batch": "auto",
            "dtype": "bf16"
        }

        hf_configs = HuggingFaceProperties(**properties)
        self.assertEqual(hf_configs.tensor_parallel_degree,
                         int(properties['tensor_parallel_degree']))
        self.assertEqual(hf_configs.cluster_size,
                         int(properties['cluster_size']))
        self.assertTrue(hf_configs.load_in_8bit)
        self.assertTrue(hf_configs.low_cpu_mem_usage)
        self.assertFalse(hf_configs.disable_flash_attn)
        self.assertEqual(hf_configs.device_map, properties['device_map'])
        self.assertTrue(hf_configs.mpi_mode)
        self.assertDictEqual(
            hf_configs.kwargs, {
                'trust_remote_code': True,
                "low_cpu_mem_usage": True,
                "device_map": 'cpu',
                "load_in_8bit": True,
                "waiting_steps": 12,
                "torch_dtype": torch.bfloat16
            })

    @mock.patch("torch.cuda.device_count")
    def test_hf_device_map(self, mock_device_count):
        mock_device_count.return_value = 4
        properties = {
            "model_id": "model_id",
            "tensor_parallel_degree": 4,
            "pipeline_parallel_degree": 1,
            "cluster_size": 1,
        }

        hf_configs = HuggingFaceProperties(**properties, rolling_batch="auto")
        self.assertEqual(hf_configs.kwargs.get("device_map"), "auto")

        hf_configs = HuggingFaceProperties(**properties,
                                           rolling_batch="disable")
        self.assertIsNone(hf_configs.kwargs.get("device_map"))

    @parameters([{
        "model_id": "model_id",
        "quantize": "bitsandbytes4"
    }, {
        "model_id": "model_id",
        "load_in_8bit": "true"
    }])
    def test_hf_error_case(self, params):
        with self.assertRaises(ValueError):
            HuggingFaceProperties(**params)

    def test_vllm_properties(self):

        def validate_vllm_config_and_engine_args_match(
            vllm_config_value,
            engine_arg_value,
            expected_value,
        ):
            self.assertEqual(vllm_config_value, expected_value)
            self.assertEqual(engine_arg_value, expected_value)

        def test_vllm_default_properties():
            required_properties = {
                "engine": "Python",
                "model_id": "some_model",
            }
            vllm_configs = VllmRbProperties(**required_properties)
            engine_args = vllm_configs.get_engine_args()
            validate_vllm_config_and_engine_args_match(
                vllm_configs.model_id_or_path, engine_args.model, "some_model")
            validate_vllm_config_and_engine_args_match(
                vllm_configs.tensor_parallel_degree,
                engine_args.tensor_parallel_size, 1)
            validate_vllm_config_and_engine_args_match(
                vllm_configs.pipeline_parallel_degree,
                engine_args.pipeline_parallel_size, 1)
            validate_vllm_config_and_engine_args_match(
                vllm_configs.quantize, engine_args.quantization, None)
            validate_vllm_config_and_engine_args_match(
                vllm_configs.max_rolling_batch_size, engine_args.max_num_seqs,
                32)
            validate_vllm_config_and_engine_args_match(vllm_configs.dtype,
                                                       engine_args.dtype,
                                                       'auto')
            validate_vllm_config_and_engine_args_match(vllm_configs.max_loras,
                                                       engine_args.max_loras,
                                                       4)
            validate_vllm_config_and_engine_args_match(
                vllm_configs.cpu_offload_gb_per_gpu,
                engine_args.cpu_offload_gb, EngineArgs.cpu_offload_gb)
            self.assertEqual(
                len(vllm_configs.get_additional_vllm_engine_args()), 0)

        def test_invalid_pipeline_parallel():
            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "tensor_parallel_degree": "4",
                "pipeline_parallel_degree": "2",
            }
            with self.assertRaises(ValueError):
                _ = VllmRbProperties(**properties)

        def test_invalid_engine():
            properties = {
                "engine": "bad_engine",
                "model_id": "some_model",
            }
            with self.assertRaises(ValueError):
                _ = VllmRbProperties(**properties)

        def test_aliases():
            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "quantization": "awq",
                "max_num_batched_tokens": "546",
                "cpu_offload_gb": "7"
            }
            vllm_configs = VllmRbProperties(**properties)
            engine_args = vllm_configs.get_engine_args()
            validate_vllm_config_and_engine_args_match(
                vllm_configs.quantize, engine_args.quantization, "awq")
            validate_vllm_config_and_engine_args_match(
                vllm_configs.max_rolling_batch_prefill_tokens,
                engine_args.max_num_batched_tokens, 546)
            validate_vllm_config_and_engine_args_match(
                vllm_configs.cpu_offload_gb_per_gpu,
                engine_args.cpu_offload_gb, 7)

        def test_long_lora_scaling_factors():
            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "long_lora_scaling_factors": "3.0"
            }
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "3"
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "3.0,4.0"
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, 4.0))

            properties['long_lora_scaling_factors'] = "3.0, 4.0 "
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, 4.0))

            properties['long_lora_scaling_factors'] = "(3.0,)"
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "(3.0,4.0)"
            vllm_props = VllmRbProperties(**properties)
            engine_args = vllm_props.get_engine_args()
            self.assertEqual(engine_args.long_lora_scaling_factors, (3.0, 4.0))

        def test_invalid_long_lora_scaling_factors():
            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "long_lora_scaling_factors": "a,b"
            }
            with self.assertRaises(ValueError):
                _ = VllmRbProperties(**properties)

        def test_conflicting_djl_vllm_conflicts():
            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "tensor_parallel_degree": 2,
                "tensor_parallel_size": 1,
            }
            vllm_configs = VllmRbProperties(**properties)
            with self.assertRaises(ValueError):
                vllm_configs.get_engine_args()

            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "pipeline_parallel_degree": 1,
                "pipeline_parallel_size": 0,
            }
            vllm_configs = VllmRbProperties(**properties)
            with self.assertRaises(ValueError):
                vllm_configs.get_engine_args()

            properties = {
                "engine": "Python",
                "model_id": "some_model",
                "max_rolling_batch_size": 1,
                "max_num_seqs": 2,
            }
            vllm_configs = VllmRbProperties(**properties)
            with self.assertRaises(ValueError):
                vllm_configs.get_engine_args()

        def test_all_vllm_engine_args():
            properties = {
                "model_id": "some_model",
                "served_model_name": "my_model",
                "skip_tokenizer_init": "true",
                "tokenizer_model": "slow",
                "trust_remote_code": "true",
                "download_dir": "model/",
                "load_format": "dummy",
                "config_format": "hf",
                "dtype": "float16",
                "kv_cache_dtype": "fp8",
                "quantization_param_path": "quant/",
                "seed": "7",
                "max_model_len": "1234",
                "worker_use_ray": "true",
                "distributed_executor_backend": "mp",
                "pipeline_parallel_degree": "1",
                "tensor_parallel_degree": "1",
                "max_parallel_loading_workers": "2",
                "block_size": "32",
                "enable_prefix_caching": "true",
                "disable_sliding_window": "true",
                "use_v2_block_manager": "false",
                "swap_space": "50",
                "cpu_offload_gb": "50",
                "gpu_memory_utilization": "0.3",
                "max_num_batched_tokens": "40",
                "max_rolling_batch_size": "10",
                "max_logprobs": "12",
                "disable_log_stats": "true",
                "revision": "asdsd",
                "code_revision": "asdfs",
                "rope_scaling": '{"scale": "1.0"}',
                "rope_theta": "0.3",
                "tokenizer_revision": "asdd",
                "quantization": "awq",
                "enforce_eager": "true",
                "max_context_len_to_capture": "1234",
                "disable_custom_all_reduce": "true",
                "tokenizer_pool_size": "12",
                "tokenizer_pool_type": "mytype",
                "tokenizer_pool_extra_config": '{"a": "b"}',
                "limit_mm_per_prompt": '{"image":2}',
                "enable_lora": "true",
                "max_loras": "5",
                "max_lora_rank": "123",
                "enable_prompt_adapter": "true",
                "max_prompt_adapters": "3",
                "max_prompt_adapter_token": "4",
                "fully_sharded_loras": "true",
                "lora_extra_vocab_size": "123",
                "long_lora_scaling_factors": "3.0",
                "lora_dtype": "float16",
                "max_cpu_loras": "320",
                "device": "cpu",
                "num_scheduler_steps": "2",
                "multi_step_stream_outputs": "false",
                "ray_workers_use_nsight": "true",
                "num_gpu_blocks_override": "4",
                "num_lookahead_slots": "4",
                "model_loader_extra_config": '{"a": "b"}',
                "ignore_patterns": "*.bin",
                "preemption_mode": "swap",
                "scheduler_delay_factor": "1.0",
                "enable_chunked_prefill": "true",
                "guided_decoding_backend": "lm-format-enforcer",
                "speculative_model": "spec_model",
                "speculative_model_quantization": "awq",
                "speculative_draft_tenosr_parallel_size": "2",
                "num_speculative_tokens": "4",
                "speculative_disable_mqa_scorer": "true",
                "speculative_max_model_len": "450",
                "speculative_disable_by_batch_size": "45",
                "ngram_prompt_lookup_max": "4",
                "ngram_prompt_lookup_min": "1",
                "spec_decoding_acceptance_method":
                "typical_acceptance_sampler",
                "typical_acceptance_sampler_posterior_threshold": "0.2",
                "typical_acceptance_sampler_posterior_alpha": "0.2",
                "qlora_adapter_name_or_path": "qlora_path/",
                "disable_logprobs_during_spec_decoding": "true",
                "otlp_traces_endpoint": "endpoint",
                "collect_detailed_traces": "yes",
                "disable_async_output_proc": "true",
                "override_neuron_config": '{"a": "b"}',
                "mm_processor_kwargs": '{"a": "b"}',
                "scheduling_policy": "priority",
            }
            vllm_configs = VllmRbProperties(**properties)
            engine_configs = vllm_configs.get_engine_args()

        # test_vllm_default_properties()
        # test_invalid_pipeline_parallel()
        # test_invalid_engine()
        # test_aliases()
        # test_long_lora_scaling_factors()
        # test_invalid_long_lora_scaling_factors()
        # test_conflicting_djl_vllm_conflicts()
        # test_all_vllm_engine_args()

    def test_sd_inf2_properties(self):
        properties = {
            'height': 128,
            'width': 128,
            'dtype': "bf16",
            'num_images_per_prompt': 2,
            'use_auth_token': 'auth_token',
            'save_mp_checkpoint_path': 'path'
        }
        properties = {**common_properties, **properties}
        neuron_sd_config = StableDiffusionNeuronXProperties(**properties)
        self.assertEqual(properties['height'], neuron_sd_config.height)
        self.assertEqual(properties['width'], neuron_sd_config.width)
        self.assertEqual(properties['use_auth_token'],
                         neuron_sd_config.use_auth_token)
        self.assertEqual(properties['save_mp_checkpoint_path'],
                         neuron_sd_config.save_mp_checkpoint_path)

    @parameters([{'height': 128, 'width': 128, "dtype": 'fp16'}])
    def test_sd_inf2_properties_errors(self, params):
        test_properties = {**common_properties, **params}
        with self.assertRaises(ValueError):
            StableDiffusionNeuronXProperties(**test_properties)


if __name__ == '__main__':
    unittest.main()
