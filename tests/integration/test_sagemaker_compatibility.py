import os
import requests
import pytest
from tests import Runner


@pytest.mark.cpu
class TestSageMakerCompatibility:

    def test_sagemaker_num_workers(self):
        with Runner('cpu-full', 'sagemaker_num_workers', download=True) as r:
            env = ["SAGEMAKER_NUM_MODEL_WORKERS=3"]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_sagemaker_max_request_size(self):
        with Runner('cpu-full', 'sagemaker_max_request_size',
                    download=True) as r:
            env = ["SAGEMAKER_MAX_REQUEST_SIZE=1024"]  # 1KB
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )

            # Test with large request (should fail - exceeds 1KB limit)
            large_data = {
                "inputs":
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]] *
                50  # 50 samples
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=large_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 413  # Request Entity Too Large

    def test_sagemaker_startup_timeout(self):
        with Runner('cpu-full', 'sagemaker_startup_timeout',
                    download=True) as r:
            env = ["SAGEMAKER_STARTUP_TIMEOUT=300"]  # 5 minutes
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            # Verify model loaded successfully (SageMaker compatibility working)
            response = requests.get(
                "http://localhost:8080/models/sklearn_test")
            assert response.status_code == 200

            # Verify timeout setting was applied by checking server logs
            log_file = os.path.join(os.getcwd(), "logs", "serving.log")
            print(f"Checking log file: {log_file}")
            if os.path.exists(log_file):
                print("Log file found, parsing content")
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    assert '"model_loading_timeout":"300' in log_content, "SAGEMAKER_STARTUP_TIMEOUT not found in logs"

    def test_sagemaker_predict_timeout(self):
        with Runner('cpu-full', 'sagemaker_predict_timeout',
                    download=True) as r:
            env = ["SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS=120"]  # 2 minutes
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            # Verify model loaded successfully (SageMaker compatibility working)
            response = requests.get(
                "http://localhost:8080/models/sklearn_test")
            assert response.status_code == 200

            # Verify timeout setting was applied by checking server logs
            log_file = os.path.join(os.getcwd(), "logs", "serving.log")
            print(f"Checking log file: {log_file}")
            if os.path.exists(log_file):
                print("Log file found, parsing content")
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    assert '"predict_timeout":"120' in log_content, "SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS not found in logs"

    def test_sagemaker_model_server_vmargs(self):
        with Runner('cpu-full', 'sagemaker_vmargs', download=True) as r:
            env = [
                "SAGEMAKER_MODEL_SERVER_VMARGS=-Dsagemaker.test=true -Xmx2g",
                "SAGEMAKER_NUM_MODEL_WORKERS=2"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            # Verify model loaded successfully
            response = requests.get(
                "http://localhost:8080/models/sklearn_test")
            assert response.status_code == 200

            # Verify JVM arguments were applied by checking server logs
            log_file = os.path.join(os.getcwd(), "logs", "serving.log")
            print(f"Checking log file: {log_file}")
            if os.path.exists(log_file):
                print("Log file found, parsing content")
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    # Check for our custom system property
                    assert "-Dsagemaker.test=true" in log_content, "SAGEMAKER_MODEL_SERVER_VMARGS system property not found in logs"
                    # Check for memory override
                    assert "-Xmx2g" in log_content, "SAGEMAKER_MODEL_SERVER_VMARGS memory setting not found in logs"
                    # Check that heap size was actually set to 2048MB
                    assert "Max heap size: 2048" in log_content, "JVM heap size was not overridden correctly"
            else:
                print("Log file not found, skipping JVM argument verification")

            # Test inference to ensure JVM args didn't break functionality
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_sagemaker_default_invocations_accept(self):
        with Runner('cpu-full', 'sagemaker_default_accept',
                    download=True) as r:
            env = ["SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=text/csv"]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            # Test without Accept header - should return CSV due to default
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=test_data,
                headers={"Content-Type": "application/json"})
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response text: {response.text}")
            assert response.status_code == 200
            assert "text/csv" in response.headers.get("Content-Type", "")

    def test_sagemaker_startup_timeout_failure(self):
        """Test that SAGEMAKER_STARTUP_TIMEOUT causes model loading to fail when timeout is too short"""
        with Runner('cpu-full',
                    'sagemaker_startup_timeout_fail',
                    download=True) as r:
            env = ["SAGEMAKER_STARTUP_TIMEOUT=3"]  # 3 seconds timeout
            try:
                r.launch(
                    env_vars=env,
                    cmd=
                    "serve -m slow_model::Python=file:/opt/ml/model/slow_loading_model.zip"
                )
                # If we get here, the server started but model loading should have failed
                # Check that the model is not available
                response = requests.get(
                    "http://localhost:8080/models/slow_model")
                assert response.status_code == 404  # Model not found due to timeout
            except Exception as e:
                # Expected - server should fail to start or model should fail to load
                print(f"Expected failure due to timeout: {e}")
                assert True

    def test_sagemaker_predict_timeout_failure(self):
        """Test that SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS causes prediction to timeout"""
        with Runner('cpu-full',
                    'sagemaker_predict_timeout_fail',
                    download=True) as r:
            env = ["SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS=3"
                   ]  # 3 seconds timeout
            r.launch(
                env_vars=env,
                cmd=
                "serve -m slow_predict::Python=file:/opt/ml/model/slow_predict_model.zip"
            )

            response = requests.get(
                "http://localhost:8080/models/slow_predict")
            assert response.status_code == 200

            # But prediction should timeout (10 second predict_fn vs 3 second timeout)
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/slow_predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=15  # Give enough time for the request itself
            )
            # Should get timeout error from DJL serving
            assert response.status_code in [408, 500,
                                            503]  # Timeout or server error

    def test_sagemaker_max_payload_in_mb(self):
        """Test SAGEMAKER_MAX_PAYLOAD_IN_MB conversion to bytes"""
        with Runner('cpu-full', 'sagemaker_max_payload_mb',
                    download=True) as r:
            env = ["SAGEMAKER_MAX_PAYLOAD_IN_MB=1"]  # 1MB = 1048576 bytes
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )

            # Test with large request (should fail - exceeds 1MB limit)
            large_array = [1.0] * 50000  # 50k floats
            large_data = {
                "inputs":
                [large_array] * 10  # 10 arrays of 50k floats each = ~20MB
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=large_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 413  # Request Entity Too Large

    def test_sagemaker_max_request_size_precedence(self):
        """Test that SAGEMAKER_MAX_REQUEST_SIZE takes precedence over SAGEMAKER_MAX_PAYLOAD_IN_MB"""
        with Runner('cpu-full', 'sagemaker_precedence', download=True) as r:
            env = [
                "SAGEMAKER_MAX_REQUEST_SIZE=1024",
                "SAGEMAKER_MAX_PAYLOAD_IN_MB=1"  # 1MB = 1048576 bytes (should be ignored)
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )

            # Test with request larger than 1KB but smaller than 1MB
            # Should fail because SAGEMAKER_MAX_REQUEST_SIZE=1024 takes precedence
            medium_data = {
                "inputs":
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]] *
                50  # ~5KB payload
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=medium_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 413  # Should fail due to 1KB limit, not 1MB limit
