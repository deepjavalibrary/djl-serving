import os
import json
import unittest
from unittest import mock

from djl_python.properties_manager.properties import Properties
from djl_python.properties_manager.tnx_properties import (
    TransformerNeuronXProperties, TnXGenerationStrategy, TnXModelSchema,
    TnXMemoryLayout, TnXDtypeName, TnXModelLoaders)
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.properties_manager.hf_properties import HuggingFaceProperties, HFQuantizeMethods
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
from djl_python.properties_manager.sd_inf2_properties import StableDiffusionNeuronXProperties
from djl_python.properties_manager.lmi_dist_rb_properties import LmiDistRbProperties, LmiDistQuantizeMethods
from djl_python.properties_manager.scheduler_rb_properties import SchedulerRbProperties
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
    }, {
        "rolling_batch": "lmi-dist",
        "is_error_case": True
    }])
    def test_trt_llm_configs(self, params):
        is_error_case = params.pop("is_error_case", False)
        properties = {**model_min_properties, **params}
        if is_error_case:
            with self.assertRaises(ValueError):
                TensorRtLlmProperties(**properties)
        else:
            trt_configs = TensorRtLlmProperties(**properties)
            self.assertEqual(trt_configs.model_id_or_path,
                             properties['model_id'])
            self.assertEqual(trt_configs.rolling_batch.value,
                             properties['rolling_batch'])

    def test_hf_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "low_cpu_mem_usage": "true",
            "disable_flash_attn": "false",
            "mpi_mode": "true",
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

        hf_configs = HuggingFaceProperties(**properties)
        self.assertIsNone(hf_configs.kwargs.get("device_map"))

    def test_hf_quantize(self):
        properties = {
            'model_id': 'model_id',
            'quantize': 'bitsandbytes8',
            'rolling_batch': 'lmi-dist'
        }
        hf_configs = HuggingFaceProperties(**properties)
        self.assertEqual(hf_configs.quantize.value,
                         HFQuantizeMethods.bitsandbytes.value)

    @parameters([{
        "model_id": "model_id",
        "quantize": HFQuantizeMethods.bitsandbytes4.value
    }, {
        "model_id": "model_id",
        "load_in_8bit": "true"
    }])
    def test_hf_error_case(self, params):
        with self.assertRaises(ValueError):
            HuggingFaceProperties(**params)

    def test_vllm_properties(self):
        # test with valid vllm properties

        def test_vllm_valid(properties):
            vllm_configs = VllmRbProperties(**properties)
            self.assertEqual(vllm_configs.model_id_or_path,
                             properties['model_id'])
            self.assertEqual(vllm_configs.engine, properties['engine'])
            self.assertEqual(
                vllm_configs.max_rolling_batch_prefill_tokens,
                int(properties['max_rolling_batch_prefill_tokens']))
            self.assertEqual(vllm_configs.dtype, properties['dtype'])
            self.assertEqual(vllm_configs.load_format,
                             properties['load_format'])
            self.assertEqual(vllm_configs.quantize, properties['quantize'])
            self.assertEqual(vllm_configs.tensor_parallel_degree,
                             int(properties['tensor_parallel_degree']))
            self.assertEqual(vllm_configs.max_model_len,
                             int(properties['max_model_len']))
            self.assertEqual(vllm_configs.enforce_eager,
                             bool(properties['enforce_eager']))
            self.assertEqual(vllm_configs.enable_lora,
                             bool(properties['enable_lora']))
            self.assertEqual(vllm_configs.gpu_memory_utilization,
                             float(properties['gpu_memory_utilization']))

        # test with invalid quantization
        def test_invalid_quantization_method(properties):
            properties['quantize'] = 'gguf'
            with self.assertRaises(ValueError):
                VllmRbProperties(**properties)
            properties['quantize'] = 'awq'

        def test_enforce_eager(properties):
            properties.pop('enforce_eager')
            properties.pop('quantize')
            self.assertTrue("enforce_eager" not in properties)
            vllm_props = VllmRbProperties(**properties)
            self.assertTrue(vllm_props.enforce_eager is False)

        def test_long_lora_scaling_factors(properties):
            properties['long_lora_scaling_factors'] = "3.0"
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "3"
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "3.0,4.0"
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, 4.0))

            properties['long_lora_scaling_factors'] = "3.0, 4.0 "
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, 4.0))

            properties['long_lora_scaling_factors'] = "(3.0,)"
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, ))

            properties['long_lora_scaling_factors'] = "(3.0,4.0)"
            vllm_props = VllmRbProperties(**properties)
            self.assertEqual(vllm_props.long_lora_scaling_factors, (3.0, 4.0))

        def test_invalid_long_lora_scaling_factors(properties):
            properties['long_lora_scaling_factors'] = "a,b"
            with self.assertRaises(ValueError):
                VllmRbProperties(**properties)

        properties = {
            'model_id': 'sample_model_id',
            'engine': 'Python',
            'max_rolling_batch_prefill_tokens': '12500',
            'max_model_len': '12800',
            'tensor_parallel_degree': '2',
            'dtype': 'fp16',
            'quantize': 'awq',
            'enforce_eager': "True",
            'enable_lora': "true",
            "gpu_memory_utilization": "0.85",
            'load_format': 'pt'
        }
        test_vllm_valid(properties.copy())
        test_invalid_quantization_method(properties.copy())
        test_enforce_eager(properties.copy())
        test_long_lora_scaling_factors(properties.copy())
        test_invalid_long_lora_scaling_factors(properties.copy())

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

    def test_lmi_dist_properties(self):

        def test_with_min_properties():
            lmi_configs = LmiDistRbProperties(**min_properties)
            self.assertEqual(lmi_configs.model_id_or_path,
                             min_properties['model_id'])
            self.assertEqual(lmi_configs.load_format, 'auto')
            self.assertEqual(lmi_configs.dtype, 'auto')
            self.assertEqual(lmi_configs.gpu_memory_utilization, 0.9)
            self.assertTrue(lmi_configs.mpi_mode)
            self.assertFalse(lmi_configs.enable_lora)

        def test_with_most_properties():
            properties = {
                'trust_remote_code': 'TRUE',
                'tensor_parallel_degree': '2',
                'revision': 'somerevisionstr',
                'max_rolling_batch_size': '64',
                'max_rolling_batch_prefill_tokens': '12500',
                'dtype': 'fp32',
                'enable_lora': "true",
            }

            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.engine, min_properties['engine'])
            self.assertEqual(lmi_configs.model_id_or_path,
                             min_properties['model_id'])
            self.assertEqual(lmi_configs.tensor_parallel_degree,
                             int(properties['tensor_parallel_degree']))
            self.assertEqual(lmi_configs.revision, properties['revision'])
            self.assertEqual(lmi_configs.max_rolling_batch_size,
                             int(properties['max_rolling_batch_size']))
            self.assertEqual(
                lmi_configs.max_rolling_batch_prefill_tokens,
                int(properties['max_rolling_batch_prefill_tokens']))
            self.assertEqual(lmi_configs.dtype, 'fp32')
            self.assertTrue(lmi_configs.mpi_mode)
            self.assertTrue(lmi_configs.trust_remote_code)
            self.assertEqual(lmi_configs.enable_lora,
                             bool(properties['enable_lora']))

        def test_invalid_quantization():
            properties = {'quantize': 'invalid'}
            with self.assertRaises(ValueError):
                LmiDistRbProperties(**properties, **min_properties)

        def test_quantization_with_dtype_error():
            # you cannot give both quantization method and dtype
            properties = {'quantize': 'bitsandbytes', 'dtype': 'int8'}
            with self.assertRaises(ValueError):
                LmiDistRbProperties(**properties, **min_properties)

        def test_quantization_squeezellm():
            properties = {'quantize': 'squeezellm'}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.quantize.value,
                             LmiDistQuantizeMethods.squeezellm.value)

        def test_long_lora_scaling_factors():
            properties = {"long_lora_scaling_factors": "3.0"}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, ))

            properties = {"long_lora_scaling_factors": "3"}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, ))

            properties = {"long_lora_scaling_factors": "3.0,4.0"}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, 4.0))

            properties = {"long_lora_scaling_factors": "3.0, 4.0 "}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, 4.0))

            properties = {"long_lora_scaling_factors": "(3.0,)"}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, ))

            properties = {"long_lora_scaling_factors": "(3.0,4.0)"}
            lmi_configs = LmiDistRbProperties(**properties, **min_properties)
            self.assertEqual(lmi_configs.long_lora_scaling_factors, (3.0, 4.0))

        def test_invalid_long_lora_scaling_factors():
            properties = {'long_lora_scaling_factors': "(a,b)"}
            with self.assertRaises(ValueError):
                LmiDistRbProperties(**properties, **min_properties)

        min_properties = {
            'engine': 'MPI',
            'mpi_mode': 'true',
            'model_id': 'sample_model_id',
        }
        test_with_min_properties()
        test_with_most_properties()
        test_invalid_quantization()
        test_quantization_with_dtype_error()
        test_quantization_squeezellm()
        test_long_lora_scaling_factors()
        test_invalid_long_lora_scaling_factors()

    def test_scheduler_properties(self):
        properties = {
            'model_id': 'sample_model_id',
            'disable_flash_attn': 'false',
            'decoding_strategy': 'beam',
            'max_sparsity': '0.44',
            'max_splits': '3',
            'multi_gpu': 'lmi_dist_sharding'
        }

        scheduler_configs = SchedulerRbProperties(**properties)
        self.assertFalse(scheduler_configs.disable_flash_attn)
        self.assertEqual(scheduler_configs.model_id_or_path,
                         properties['model_id'])
        self.assertEqual(scheduler_configs.decoding_strategy,
                         properties['decoding_strategy'])
        self.assertEqual(scheduler_configs.max_sparsity,
                         float(properties['max_sparsity']))
        self.assertEqual(scheduler_configs.max_splits,
                         int(properties['max_splits']))
        self.assertEqual(scheduler_configs.multi_gpu, properties['multi_gpu'])


if __name__ == '__main__':
    unittest.main()
