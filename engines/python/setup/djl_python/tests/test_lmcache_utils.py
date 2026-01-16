import os
import tempfile
import json
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

from djl_python.lmcache_utils import (
    apply_lmcache_auto_config,
    calculate_cpu_cache_size,
    calculate_disk_cache_size,
    calculate_model_size_from_hf_api,
    get_available_cpu_memory,
    get_directory_size_gb,
    get_model_size_gb,
    set_lmcache_env_vars,
)


class TestLMCacheUtils(unittest.TestCase):

    def setUp(self):
        """Clear environment variables before each test"""
        # Store original env vars
        self.original_env = os.environ.copy()
        # Clear any LMCACHE_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("LMCACHE_"):
                del os.environ[key]

    def tearDown(self):
        """Restore original environment after each test"""
        os.environ.clear()
        os.environ.update(self.original_env)

    # ========== Fast-fail validation tests ==========

    def test_auto_config_disabled_returns_unchanged_properties(self):
        """Test that auto-config does nothing when disabled"""
        properties = {"model_id": "test-model", "lmcache_auto_config": "false"}
        result = apply_lmcache_auto_config("/fake/path", properties)
        self.assertEqual(result, properties)

    def test_auto_config_missing_returns_unchanged_properties(self):
        """Test that auto-config does nothing when not specified"""
        properties = {"model_id": "test-model"}
        result = apply_lmcache_auto_config("/fake/path", properties)
        self.assertEqual(result, properties)

    def test_fails_with_expert_parallelism_enabled(self):
        """Test that auto-config fails fast when expert parallelism is enabled"""
        properties = {
            "lmcache_auto_config": "true",
            "enable_expert_parallel": "true"
        }
        with self.assertRaises(RuntimeError) as context:
            apply_lmcache_auto_config("/fake/path", properties)

        self.assertIn("expert parallelism", str(context.exception).lower())
        self.assertIn("enable_expert_parallel=true", str(context.exception))

    def test_warns_with_existing_lmcache_env_vars(self):
        """Test that auto-config warns but continues when LMCACHE env vars exist"""
        os.environ["LMCACHE_CONFIG_FILE"] = "/some/config.yaml"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "100"

        properties = {"lmcache_auto_config": "true"}

        # Mock the dependencies so we can test the warning behavior
        with patch('djl_python.lmcache_utils.get_model_size_gb') as mock_size, \
             patch('djl_python.lmcache_utils.get_available_cpu_memory') as mock_cpu, \
             patch('djl_python.lmcache_utils.shutil.disk_usage') as mock_disk, \
             patch('djl_python.lmcache_utils.logger') as mock_logger:

            mock_size.return_value = 7.0
            mock_cpu.return_value = 100.0
            mock_stat = MagicMock()
            mock_stat.total = 500 * 1024**3
            mock_disk.return_value = mock_stat

            # Should not raise, but should warn
            result = apply_lmcache_auto_config("/fake/path", properties)

            # Check that warning was logged
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("existing LMCACHE environment variables",
                          warning_call)
            self.assertIn("will be overwritten", warning_call)

            # Check that config was still applied
            self.assertIn("kv_transfer_config", result)

    def test_fails_with_manual_config_file_property(self):
        """Test that auto-config fails fast when lmcache_config_file is set"""
        properties = {
            "lmcache_auto_config": "true",
            "lmcache_config_file": "/path/to/config.yaml"
        }

        with self.assertRaises(RuntimeError) as context:
            apply_lmcache_auto_config("/fake/path", properties)

        error_msg = str(context.exception)
        self.assertIn("lmcache_config_file", error_msg)
        self.assertIn("/path/to/config.yaml", error_msg)
        self.assertIn("incompatible with manual LMCache configuration",
                      error_msg)

    def test_fails_only_with_config_file_property(self):
        """Test that auto-config only fails with config file property, not env vars"""
        # Set env vars (should only warn)
        os.environ["LMCACHE_LOCAL_DISK"] = "/tmp/lmcache"

        # But also set property (should fail)
        properties = {
            "lmcache_auto_config": "true",
            "lmcache_config_file": "/config.yaml"
        }

        with self.assertRaises(RuntimeError) as context:
            apply_lmcache_auto_config("/fake/path", properties)

        error_msg = str(context.exception)
        # Should mention the property
        self.assertIn("lmcache_config_file", error_msg)
        # Should NOT mention env vars in the error (they just get a warning)
        self.assertNotIn("LMCACHE_LOCAL_DISK", error_msg)

    # ========== Model size calculation tests ==========

    @patch('djl_python.lmcache_utils.HfApi')
    def test_model_size_from_hf_api_with_safetensors(self, mock_hf_api):
        """Test model size calculation from HF API with safetensors files"""
        # Mock HF API response
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        mock_sibling1 = MagicMock()
        mock_sibling1.rfilename = "model-00001-of-00002.safetensors"
        mock_sibling1.size = 5 * 1024**3  # 5GB

        mock_sibling2 = MagicMock()
        mock_sibling2.rfilename = "model-00002-of-00002.safetensors"
        mock_sibling2.size = 3 * 1024**3  # 3GB

        mock_info = MagicMock()
        mock_info.siblings = [mock_sibling1, mock_sibling2]
        mock_api_instance.model_info.return_value = mock_info

        size_gb = calculate_model_size_from_hf_api("test-org/test-model")

        self.assertAlmostEqual(size_gb, 8.0, places=1)
        mock_api_instance.model_info.assert_called_once_with(
            "test-org/test-model", files_metadata=True)

    @patch('djl_python.lmcache_utils.HfApi')
    def test_model_size_from_hf_api_with_bin_files(self, mock_hf_api):
        """Test model size calculation from HF API with .bin files"""
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        mock_sibling = MagicMock()
        mock_sibling.rfilename = "pytorch_model.bin"
        mock_sibling.size = 7 * 1024**3  # 7GB

        mock_info = MagicMock()
        mock_info.siblings = [mock_sibling]
        mock_api_instance.model_info.return_value = mock_info

        size_gb = calculate_model_size_from_hf_api("test-org/test-model")

        self.assertAlmostEqual(size_gb, 7.0, places=1)

    def test_get_directory_size_with_safetensors(self):
        """Test directory size calculation with safetensors files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "model.safetensors")
            file2 = os.path.join(tmpdir, "model-2.safetensors")

            with open(file1, 'wb') as f:
                f.write(b'0' * (2 * 1024**3))  # 2GB
            with open(file2, 'wb') as f:
                f.write(b'0' * (3 * 1024**3))  # 3GB

            size_gb = get_directory_size_gb(tmpdir)
            self.assertAlmostEqual(size_gb, 5.0, places=1)

    def test_get_directory_size_with_bin_files(self):
        """Test directory size calculation with .bin files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "pytorch_model.bin")

            with open(file1, 'wb') as f:
                f.write(b'0' * (4 * 1024**3))  # 4GB

            size_gb = get_directory_size_gb(tmpdir)
            self.assertAlmostEqual(size_gb, 4.0, places=1)

    @patch('djl_python.lmcache_utils.calculate_model_size_from_hf_api')
    def test_get_model_size_with_hf_id(self, mock_hf_api):
        """Test get_model_size_gb with HuggingFace model ID"""
        mock_hf_api.return_value = 7.5

        size_gb = get_model_size_gb("meta-llama/Llama-2-7b-hf")

        self.assertEqual(size_gb, 7.5)
        mock_hf_api.assert_called_once_with("meta-llama/Llama-2-7b-hf")

    @patch('djl_python.lmcache_utils.get_directory_size_gb')
    @patch('os.path.isdir')
    def test_get_model_size_with_local_path(self, mock_isdir,
                                            mock_get_dir_size):
        """Test get_model_size_gb with local directory path"""
        mock_isdir.return_value = True
        mock_get_dir_size.return_value = 13.2

        size_gb = get_model_size_gb("/local/models/my-model")

        self.assertEqual(size_gb, 13.2)
        mock_get_dir_size.assert_called_once_with("/local/models/my-model")

    # ========== Cache size calculation tests ==========

    @patch('djl_python.lmcache_utils.get_available_cpu_memory')
    def test_calculate_cpu_cache_size_basic(self, mock_cpu_mem):
        """Test CPU cache size calculation with basic parameters"""
        mock_cpu_mem.return_value = 100.0  # 100GB available

        properties = {"tensor_parallel_degree": "2"}
        model_size_gb = 10.0

        cpu_cache_gb = calculate_cpu_cache_size(properties, model_size_gb)

        # Reserved: max(2*10, 0.2*100) = max(20, 20) = 20GB
        # Available: 100 - 20 = 80GB
        # Per GPU (TP=2): 80 / 2 = 40GB
        self.assertAlmostEqual(cpu_cache_gb, 40.0, places=1)

    @patch('djl_python.lmcache_utils.get_available_cpu_memory')
    def test_calculate_cpu_cache_size_large_model(self, mock_cpu_mem):
        """Test CPU cache with large model (2x model > 20% memory)"""
        mock_cpu_mem.return_value = 200.0  # 200GB available

        properties = {"tensor_parallel_degree": "4"}
        model_size_gb = 50.0  # Large model

        cpu_cache_gb = calculate_cpu_cache_size(properties, model_size_gb)

        # Reserved: max(2*50, 0.2*200) = max(100, 40) = 100GB
        # Available: 200 - 100 = 100GB
        # Per GPU (TP=4): 100 / 4 = 25GB
        self.assertAlmostEqual(cpu_cache_gb, 25.0, places=1)

    @patch('djl_python.lmcache_utils.shutil.disk_usage')
    def test_calculate_disk_cache_size_basic(self, mock_disk_usage):
        """Test disk cache size calculation with basic parameters"""
        mock_stat = MagicMock()
        mock_stat.total = 500 * 1024**3  # 500GB total
        mock_disk_usage.return_value = mock_stat

        properties = {"tensor_parallel_degree": "2"}
        model_size_gb = 20.0

        disk_cache_gb = calculate_disk_cache_size(properties, model_size_gb)

        # Reserved: max(2*20, 0.2*500) = max(40, 100) = 100GB
        # Available: 500 - 100 = 400GB
        # Per GPU (TP=2): 400 / 2 = 200GB
        self.assertAlmostEqual(disk_cache_gb, 200.0, places=1)

    @patch('djl_python.lmcache_utils.shutil.disk_usage')
    def test_calculate_disk_cache_size_large_model(self, mock_disk_usage):
        """Test disk cache with large model (2x model > 20% disk)"""
        mock_stat = MagicMock()
        mock_stat.total = 1000 * 1024**3  # 1000GB total
        mock_disk_usage.return_value = mock_stat

        properties = {"tensor_parallel_degree": "4"}
        model_size_gb = 150.0  # Large model

        disk_cache_gb = calculate_disk_cache_size(properties, model_size_gb)

        # Reserved: max(2*150, 0.2*1000) = max(300, 200) = 300GB
        # Available: 1000 - 300 = 700GB
        # Per GPU (TP=4): 700 / 4 = 175GB
        self.assertAlmostEqual(disk_cache_gb, 175.0, places=1)

    # ========== Environment variable setting tests ==========

    def test_set_lmcache_env_vars(self):
        """Test that LMCache environment variables are set correctly"""
        set_lmcache_env_vars(50.0, 200.0)

        self.assertEqual(os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"], "50")
        self.assertEqual(os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"], "200")
        self.assertEqual(os.environ["LMCACHE_LOCAL_DISK"],
                         "file:///tmp/lmcache/")
        self.assertEqual(os.environ["LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR"],
                         "true")
        self.assertEqual(os.environ["PYTHONHASHSEED"], "0")
        self.assertEqual(os.environ["LMCACHE_EXTRA_CONFIG"],
                         '{"use_odirect": true}')

    # ========== Full tests ==========

    @patch('djl_python.lmcache_utils.get_model_size_gb')
    @patch('djl_python.lmcache_utils.get_available_cpu_memory')
    @patch('djl_python.lmcache_utils.shutil.disk_usage')
    def test_auto_config_with_hf_model_id(self, mock_disk, mock_cpu_mem,
                                          mock_model_size):
        """Test full auto-config flow with HuggingFace model ID"""
        mock_model_size.return_value = 7.0
        mock_cpu_mem.return_value = 100.0
        mock_stat = MagicMock()
        mock_stat.total = 500 * 1024**3
        mock_disk.return_value = mock_stat

        properties = {
            "lmcache_auto_config": "true",
            "tensor_parallel_degree": "2"
        }

        result = apply_lmcache_auto_config("meta-llama/Llama-2-7b-hf",
                                           properties)

        # Check kv_transfer_config was added
        self.assertIn("kv_transfer_config", result)
        kv_config = json.loads(result["kv_transfer_config"])
        self.assertEqual(kv_config["kv_connector"], "LMCacheConnectorV1")
        self.assertEqual(kv_config["kv_role"], "kv_both")

        # Check env vars were set
        self.assertIn("LMCACHE_MAX_LOCAL_CPU_SIZE", os.environ)
        self.assertIn("LMCACHE_MAX_LOCAL_DISK_SIZE", os.environ)

    @patch('djl_python.lmcache_utils.get_directory_size_gb')
    @patch('djl_python.lmcache_utils.get_available_cpu_memory')
    @patch('djl_python.lmcache_utils.shutil.disk_usage')
    @patch('os.path.isdir')
    def test_auto_config_with_local_path(self, mock_isdir, mock_disk,
                                         mock_cpu_mem, mock_dir_size):
        """Test full auto-config flow with local model path"""
        mock_isdir.return_value = True
        mock_dir_size.return_value = 13.0
        mock_cpu_mem.return_value = 150.0
        mock_stat = MagicMock()
        mock_stat.total = 1000 * 1024**3
        mock_disk.return_value = mock_stat

        properties = {
            "lmcache_auto_config": "true",
            "tensor_parallel_degree": "4"
        }

        result = apply_lmcache_auto_config("/local/models/my-model",
                                           properties)

        # Check kv_transfer_config was added
        self.assertIn("kv_transfer_config", result)

        # Check env vars were set
        self.assertIn("LMCACHE_MAX_LOCAL_CPU_SIZE", os.environ)
        self.assertIn("LMCACHE_MAX_LOCAL_DISK_SIZE", os.environ)

        # Verify model size was calculated from directory
        mock_dir_size.assert_called_once_with("/local/models/my-model")


if __name__ == '__main__':
    unittest.main()
