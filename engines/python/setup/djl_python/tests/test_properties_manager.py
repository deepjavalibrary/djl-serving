import unittest
from djl_python.properties_manager.properties import Properties
from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties
from djl_python.properties_manager.trt_properties import TensorRtLlmProperties
from djl_python.properties_manager.ds_properties import DeepSpeedProperties, DsQuantizeMethods
from djl_python.properties_manager.hf_properties import HuggingFaceProperties, HFQuantizeMethods

import torch

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
    'batch_size': 4,
    'max_rolling_batch_size': 2,
    'enable_streaming': 'False'
}


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

    def test_common_configs_error_case(self):
        other_properties = min_common_properties
        other_properties["rolling_batch"] = "disable"
        other_properties["enable_streaming"] = "true"
        other_properties["batch_size"] = 2
        with self.assertRaises(ValueError):
            Properties(**other_properties)

    def test_tnx_configs(self):
        properties = {
            "n_positions": "256",
            "load_split_model": "true",
            "quantize": "bitsandbytes8",
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

    def test_tnx_configs_error_case(self):
        properties = {
            "n_positions": "256",
            "load_split_model": "true",
            "quantize": "bitsandbytes8",
        }

        def test_url_not_s3_uri(url):
            properties['compiled_graph_path'] = url
            with self.assertRaises(ValueError):
                TransformerNeuronXProperties(**common_properties, **properties)
            del properties['compiled_graph_path']

        def test_non_existent_directory(directory):
            properties['compiled_graph_path'] = directory
            with self.assertRaises(ValueError):
                TransformerNeuronXProperties(**common_properties, **properties)
            del properties['compiled_graph_path']

        test_url_not_s3_uri("https://random.url.address/")
        test_non_existent_directory("not_a_directory")

    def test_trtllm_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "rolling_batch": "auto",
        }
        trt_configs = TensorRtLlmProperties(**properties)
        self.assertEqual(trt_configs.model_id_or_path, properties['model_id'])
        self.assertEqual(trt_configs.rolling_batch.value,
                         properties['rolling_batch'])

    def test_trtllm_error_cases(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
        }

        def test_trtllm_rb_disable():
            properties['rolling_batch'] = 'disable'
            with self.assertRaises(ValueError):
                TensorRtLlmProperties(**properties)

        def test_trtllm_rb_invalid():
            properties['rolling_batch'] = 'lmi-dist'
            with self.assertRaises(ValueError):
                TensorRtLlmProperties(**properties)

        test_trtllm_rb_invalid()
        test_trtllm_rb_disable()

    def test_ds_properties(self):
        ds_properties = {
            'quantize': "dynamic_int8",
            'dtype': 'fp16',
            'max_tokens': "2048",
            'task': 'fill-mask',
            'low_cpu_mem_usage': "false",
            'enable_cuda_graph': "True",
            'triangular_masking': "false",
            'checkpoint': 'ml/model',
            'save_mp_checkpoint_path': '/opt/ml/model'
        }

        def test_ds_basic_configs():
            ds_configs = DeepSpeedProperties(**ds_properties,
                                             **common_properties)
            self.assertEqual(ds_configs.quantize.value,
                             DsQuantizeMethods.dynamicint8.value)
            self.assertEqual(ds_configs.dtype, torch.float16)
            self.assertEqual(ds_configs.max_tokens, 2048)
            self.assertEqual(ds_configs.task, ds_properties['task'])
            self.assertEqual(ds_configs.device, 0)
            self.assertFalse(ds_configs.low_cpu_mem_usage)
            self.assertTrue(ds_configs.enable_cuda_graph)
            self.assertFalse(ds_configs.triangular_masking)

            ds_config = {
                'tensor_parallel': {
                    'tp_size': 4
                },
                'enable_cuda_graph': True,
                'triangular_masking': False,
                'return_tuple': True,
                'training_mp_size': 1,
                'max_tokens': 2048,
                'base_dir': 'model_id',
                'checkpoint': 'model_id/ml/model',
                'save_mp_checkpoint_path': '/opt/ml/model',
                'dynamic_quant': {
                    'enabled': True,
                    'use_cutlass': False
                }
            }

            self.assertDictEqual(ds_configs.ds_config, ds_config)

        def test_ds_smoothquant_configs():
            ds_properties['quantize'] = 'smoothquant'
            ds_configs = DeepSpeedProperties(**ds_properties,
                                             **common_properties)
            self.assertEqual(ds_configs.quantize.value,
                             DsQuantizeMethods.smoothquant.value)
            self.assertDictEqual(ds_configs.ds_config['dynamic_quant'], {
                'enabled': True,
                'use_cutlass': False
            })
            self.assertDictEqual(ds_configs.ds_config['smoothing'], {
                'smooth': True,
                'calibrate': True
            })
            self.assertDictEqual(ds_configs.ds_config['tensor_parallel'],
                                 {'tp_size': 4})

        def test_ds_invalid_quant_method():
            ds_properties['quantize'] = 'invalid'
            ds_configs = DeepSpeedProperties(**ds_properties,
                                             **common_properties)
            self.assertIsNone(ds_configs.quantize)

        test_ds_basic_configs()
        test_ds_smoothquant_configs()
        test_ds_invalid_quant_method()

    def test_ds_error_properties(self):
        ds_properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "quantize": "smoothquant",
        }

        def test_ds_invalid_quant():
            ds_properties['dtype'] = 'bf16'
            with self.assertRaises(ValueError):
                DeepSpeedProperties(**ds_properties)

        def test_ds_invalid_sq_value():
            ds_properties['smoothquant_alpha'] = 1.5
            with self.assertRaises(ValueError):
                DeepSpeedProperties(**ds_properties)

        def test_ds_invalid_dtype():
            ds_properties['dtype'] = 'invalid'
            with self.assertRaises(ValueError):
                DeepSpeedProperties(**ds_properties)

        test_ds_invalid_quant()
        test_ds_invalid_sq_value()
        test_ds_invalid_dtype()

    def test_hf_configs(self):
        properties = {
            "model_id": "model_id",
            "model_dir": "model_dir",
            "tensor_parallel_degree": "4",
            "load_in_8bit": "true",
            "low_cpu_mem_usage": "True",
            "disable_flash_attn": "false",
            "engine": "MPI",
            "device_map": "auto"
        }

        hf_configs = HuggingFaceProperties(**properties)
        self.assertTrue(hf_configs.load_in_8bit)
        self.assertTrue(hf_configs.low_cpu_mem_usage)
        self.assertFalse(hf_configs.disable_flash_attn)
        self.assertEqual(hf_configs.device_map, properties['device_map'])
        self.assertTrue(hf_configs.is_mpi)
        self.assertDictEqual(
            hf_configs.kwargs, {
                'trust_remote_code': False,
                "low_cpu_mem_usage": True,
                "device_map": 'auto',
                "load_in_8bit": True
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

    def test_hf_error_case(self):
        properties = {"model_id": "model_id", 'load_in_8bit': True}
        with self.assertRaises(ValueError):
            HuggingFaceProperties(**properties)

        properties = {"quantize": HFQuantizeMethods.bitsandbytes4.value}
        with self.assertRaises(ValueError):
            HuggingFaceProperties(**properties)


if __name__ == '__main__':
    unittest.main()
