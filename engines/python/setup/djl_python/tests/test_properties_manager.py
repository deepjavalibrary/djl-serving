import os
import json
import unittest
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
from djl_python.chat_completions.chat_properties import ChatProperties
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

chat_messages_properties = {
    "messages": [
        {
            "role": "system",
            "content": "You are a friendly chatbot"
        },
        {
            "role": "user",
            "content": "Which is bigger, the moon or the sun?"
        },
    ]
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
            "fuse_qkv": "true",
            "enable_saturate_infinity": "true",
            "rolling_batch_strategy": "continuous_batching",
            "collectives_layout": "HSB",
            "on_device_embedding": "true",
            "partition_schema": "legacy",
            "attention_layout": "HSB",
            "cache_layout": "SBH",
            "all_reduce_dtype": "float32",
            "cast_logits_dtype": "float32",
        }
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
        self.assertTrue(tnx_configs.group_query_attention,
                        properties['group_query_attention'])
        self.assertTrue(tnx_configs.fuse_qkv)
        self.assertEqual(tnx_configs.rolling_batch_strategy,
                         TnXGenerationStrategy.continuous_batching)
        self.assertEqual(tnx_configs.collectives_layout,
                         TnXMemoryLayout.LAYOUT_HSB)
        self.assertTrue(tnx_configs.on_device_embedding)
        self.assertEqual(tnx_configs.partition_schema, TnXModelSchema.legacy)
        self.assertEqual(tnx_configs.attention_layout,
                         TnXMemoryLayout.LAYOUT_HSB)
        self.assertEqual(tnx_configs.cache_layout, TnXMemoryLayout.LAYOUT_SBH)
        self.assertEqual(tnx_configs.all_reduce_dtype, TnXDtypeName.float32)
        self.assertEqual(tnx_configs.cast_logits_dtype, TnXDtypeName.float32)
        self.assertEqual(tnx_configs.model_loader, TnXModelLoaders.tnx)

        # tests context length estimate as integer
        def test_tnx_cle_int(context_length_estimate):
            properties['context_length_estimate'] = context_length_estimate
            configs = TransformerNeuronXProperties(**common_properties,
                                                   **properties)
            self.assertEqual(configs.context_length_estimate, [256])
            del properties['context_length_estimate']

        test_tnx_cle_int('256')

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
        'partition_schema': 'optimum',
        'load_split_model': 'true'
    }, {
        'partition_schema': 'legacy',
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
        self.assertTrue(hf_configs.is_mpi)
        self.assertDictEqual(hf_configs.kwargs, {
            'trust_remote_code': False,
            "low_cpu_mem_usage": True,
        })

    def test_hf_all_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "tensor_parallel_degree": "4",
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
        self.assertTrue(hf_configs.load_in_8bit)
        self.assertTrue(hf_configs.low_cpu_mem_usage)
        self.assertFalse(hf_configs.disable_flash_attn)
        self.assertEqual(hf_configs.device_map, properties['device_map'])
        self.assertTrue(hf_configs.is_mpi)
        self.assertDictEqual(
            hf_configs.kwargs, {
                'trust_remote_code': True,
                "low_cpu_mem_usage": True,
                "device_map": 'cpu',
                "load_in_8bit": True,
                "waiting_steps": 12,
                "torch_dtype": torch.bfloat16
            })

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

        properties = {
            'model_id': 'sample_model_id',
            'engine': 'Python',
            'max_rolling_batch_prefill_tokens': '12500',
            'max_model_len': '12800',
            'tensor_parallel_degree': '2',
            'dtype': 'fp16',
            'quantize': 'awq',
            'enforce_eager': "True",
            "gpu_memory_utilization": "0.85",
            'load_format': 'pt'
        }
        test_vllm_valid(properties.copy())
        test_invalid_quantization_method(properties.copy())
        test_enforce_eager(properties.copy())

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
            self.assertTrue(lmi_configs.is_mpi)

        def test_with_most_properties():
            properties = {
                'trust_remote_code': 'TRUE',
                'tensor_parallel_degree': '2',
                'revision': 'somerevisionstr',
                'max_rolling_batch_size': '64',
                'max_rolling_batch_prefill_tokens': '12500',
                'dtype': 'fp32',
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
            self.assertTrue(lmi_configs.is_mpi)
            self.assertTrue(lmi_configs.trust_remote_code)

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

    def test_chat_configs(self):

        def test_chat_min_configs():
            chat_configs = ChatProperties(**chat_messages_properties)
            self.assertEqual(chat_configs.messages,
                             chat_messages_properties["messages"])
            self.assertIsNone(chat_configs.model)
            self.assertEqual(chat_configs.frequency_penalty, 0.0)
            self.assertIsNone(chat_configs.logit_bias)
            self.assertFalse(chat_configs.logprobs)
            self.assertIsNone(chat_configs.top_logprobs)
            self.assertIsNone(chat_configs.max_tokens)
            self.assertEqual(chat_configs.n, 1)
            self.assertEqual(chat_configs.presence_penalty, 0.0)
            self.assertIsNone(chat_configs.seed)
            self.assertIsNone(chat_configs.stop)
            self.assertFalse(chat_configs.stream)
            self.assertEqual(chat_configs.temperature, 1.0)
            self.assertEqual(chat_configs.top_p, 1.0)
            self.assertIsNone(chat_configs.user)

        def test_chat_all_configs():
            properties = dict(chat_messages_properties)
            properties["model"] = "model"
            properties["frequency_penalty"] = "1.0"
            properties["logit_bias"] = {"2435": -100.0, "640": -100.0}
            properties["logprobs"] = "false"
            properties["top_logprobs"] = "3"
            properties["max_tokens"] = "256"
            properties["n"] = "1"
            properties["presence_penalty"] = "1.0"
            properties["seed"] = "123"
            properties["stop"] = ["stop"]
            properties["stream"] = "true"
            properties["temperature"] = "1.0"
            properties["top_p"] = "3.0"
            properties["user"] = "user"

            chat_configs = ChatProperties(**properties)
            self.assertEqual(chat_configs.messages, properties["messages"])
            self.assertEqual(chat_configs.model, properties['model'])
            self.assertEqual(chat_configs.frequency_penalty,
                             float(properties['frequency_penalty']))
            self.assertEqual(chat_configs.logit_bias, properties['logit_bias'])
            self.assertFalse(chat_configs.logprobs)
            self.assertIsNone(chat_configs.top_logprobs)
            self.assertEqual(chat_configs.max_tokens,
                             int(properties['max_tokens']))
            self.assertEqual(chat_configs.n, int(properties['n']))
            self.assertEqual(chat_configs.presence_penalty,
                             float(properties['presence_penalty']))
            self.assertEqual(chat_configs.seed, int(properties['seed']))
            self.assertEqual(chat_configs.stop, properties['stop'])
            self.assertTrue(chat_configs.stream)
            self.assertEqual(chat_configs.temperature,
                             float(properties['temperature']))
            self.assertEqual(chat_configs.top_p, float(properties['top_p']))
            self.assertEqual(chat_configs.user, properties['user'])

        test_chat_min_configs()
        test_chat_all_configs()

    @parameters([{
        "messages": [{
            "role1": "system",
            "content": "You are a friendly chatbot"
        }]
    }, {
        "frequency_penalty": "-3.0"
    }, {
        "frequency_penalty": "3.0"
    }, {
        "logit_bias": {
            "2435": -100.0,
            "640": 200.0
        }
    }, {
        "logit_bias": {
            "2435": -200.0,
            "640": 100.0
        }
    }, {
        "logprobs": "true",
        "top_logprobs": "-1"
    }, {
        "logprobs": "true",
        "top_logprobs": "30"
    }, {
        "presence_penalty": "-3.0"
    }, {
        "presence_penalty": "3.0"
    }, {
        "temperature": "-1.0"
    }, {
        "temperature": "3.0"
    }])
    def test_chat_invalid_configs(self, params):
        test_properties = {**chat_messages_properties, **params}
        with self.assertRaises(ValueError):
            ChatProperties(**test_properties)


if __name__ == '__main__':
    unittest.main()
