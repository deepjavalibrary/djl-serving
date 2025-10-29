import os
import requests
import tempfile
import shutil
import zipfile
import pytest
from tests import Runner


@pytest.mark.cpu
class TestCustomFormatters:

    def test_sklearn_all_custom_formatters(self):
        """Test sklearn handler with all four custom formatters"""
        with Runner('cpu-full', 'sklearn_custom_formatters',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_custom::Python=file:/opt/ml/model/sklearn_custom_model_sm_v2.zip"
            )

            # Test custom formatters
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_custom",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert result["probability"] == 0.999  # Custom predict formatter
            assert "model_type" in result  # Custom output formatter

    def test_sagemaker_env_with_custom_formatters(self):
        """Test SageMaker compatibility with custom formatters and env variables"""
        with Runner('cpu-full', 'sagemaker_custom_formatters',
                    download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=2",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=application/json"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sagemaker_custom::Python=file:/opt/ml/model/sklearn_custom_model_sm_v2.zip"
            )

            # Test with custom formatters - use features format from existing model
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sagemaker_custom",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 200
            result = response.json()
            assert result["probability"] == 0.999  # Custom predict formatter
            assert "model_type" in result  # Custom output formatter

    def test_sagemaker_csv_default_with_json_only_formatter(self):
        """Test SageMaker with CSV default but JSON-only output formatter (should fail)"""
        with Runner('cpu-full', 'sagemaker_csv_default', download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=1",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=text/csv"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sagemaker_custom::Python=file:/opt/ml/model/sklearn_custom_model_sm_v2.zip"
            )

            # Test should fail because output_fn only supports application/json
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sagemaker_custom",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 424  # Failed dependency - output formatter error

    def test_sagemaker_input_output_formatters(self):
        """Test input_fn and output_fn formatters -- should work without other two functions and also work
        with default predict logic in handler"""
        with Runner('cpu-full', 'sagemaker_input_output', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_io::Python=file:/opt/ml/model/sklearn_custom_model_input_output_v2.zip"
            )

            # Test with SageMaker input/output formatters
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_io",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "original_prediction" in result
            assert "doubled_prediction" in result
            assert "prediction_sum" in result
            assert result["sagemaker_output_fn_used"] == True
            # Verify doubled prediction is actually double the original
            original = result["original_prediction"][0]
            doubled = result["doubled_prediction"][0]
            assert doubled == original * 2

    def test_sagemaker_invalid_input_formatter(self):
        """Test SageMaker input_fn that returns invalid format (should fail with default handler predict)"""
        with Runner('cpu-full', 'sagemaker_invalid_input', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_invalid::Python=file:/opt/ml/model/sklearn_custom_model_input_output_invalid_v2.zip"
            )

            # Test should fail because input_fn returns raw list instead of numpy array
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_invalid",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424  # Failed dependency - input processing error

    def test_xgboost_all_sagemaker_formatters(self):
        """Test XGBoost handler with all four SageMaker formatters"""
        with Runner('cpu-full', 'xgboost_sagemaker_all', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_all::Python=file:/opt/ml/model/xgboost_sagemaker_all.zip"
            )

            # Test SageMaker formatters
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_all",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.888  # Custom predict formatter
            assert result["custom_xgb"] == True  # Custom output formatter
            assert "model_type" in result

    def test_xgboost_sagemaker_env_with_formatters(self):
        """Test XGBoost SageMaker compatibility with env variables"""
        with Runner('cpu-full', 'xgboost_sagemaker_env', download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=2",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=application/json"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m xgboost_env::Python=file:/opt/ml/model/xgboost_sagemaker_all.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_env",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.888
            assert result["custom_xgb"] == True

    def test_xgboost_csv_default_with_json_only_formatter(self):
        """Test XGBoost SageMaker with CSV default but JSON-only output formatter (should fail)"""
        with Runner('cpu-full', 'xgboost_csv_default', download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=1",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=text/csv"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m xgboost_csv::Python=file:/opt/ml/model/xgboost_sagemaker_all.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_csv",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 424  # Failed dependency - output formatter error

    def test_xgboost_sagemaker_input_output_formatters(self):
        """Test XGBoost SageMaker input_fn and output_fn formatters"""
        with Runner('cpu-full', 'xgboost_input_output', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_io::Python=file:/opt/ml/model/xgboost_sagemaker_input_output.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_io",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "original_prediction" in result
            assert "doubled_prediction" in result
            assert "prediction_sum" in result
            assert result["sagemaker_output_fn_used"] == True
            # Verify doubled prediction is actually double the original
            original = result["original_prediction"][0]
            doubled = result["doubled_prediction"][0]
            assert doubled == original * 2

    def test_xgboost_sagemaker_invalid_input_formatter(self):
        """Test XGBoost SageMaker input_fn that returns invalid format (should fail)"""
        with Runner('cpu-full', 'xgboost_invalid_input', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_invalid::Python=file:/opt/ml/model/xgboost_sagemaker_input_output_invalid.zip"
            )

            # Test should fail because input_fn returns raw list instead of numpy array
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_invalid",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424  # Failed dependency - input processing error

    def test_sklearn_mixed_djl_sagemaker_formatters(self):
        """Test sklearn with mixed DJL and SageMaker formatters - DJL should take precedence and SageMaker functions should be ignored"""
        with Runner('cpu-full', 'sklearn_mixed_formatters',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_mixed::Python=file:/opt/ml/model/sklearn_mixed_djl_sagemaker_v2.zip"
            )

            # When DJL decorators are present, SageMaker functions should be completely ignored
            # This means only DJL model loader will be used, and default input/predict/output processing will be used
            test_data = {
                "inputs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_mixed",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            # Should get default sklearn prediction output format, not custom SageMaker output
            assert "predictions" in result  # Default output format
            assert isinstance(result["predictions"], list)
            # Verify the model was loaded with DJL decorator (it has djl_loaded=True attribute)

    def test_xgboost_mixed_djl_sagemaker_formatters(self):
        """Test xgboost with mixed DJL and SageMaker formatters - DJL should take precedence and SageMaker functions should be ignored"""
        with Runner('cpu-full', 'xgboost_mixed_formatters',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_mixed::Python=file:/opt/ml/model/xgboost_mixed_djl_sagemaker_v2.zip"
            )

            # When DJL decorators are present, SageMaker functions should be completely ignored
            # This means only DJL model loader will be used, and default input/predict/output processing will be used
            test_data = {
                "inputs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_mixed",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            # Should get default xgboost prediction output format, not custom SageMaker output
            assert "predictions" in result  # Default output format
            assert isinstance(result["predictions"], list)
            # Verify the model was loaded with DJL decorator (it has djl_loaded=True attribute)

    def test_sklearn_djl_all_formatters(self):
        """Test sklearn handler with all four DJL decorators"""
        with Runner('cpu-full', 'sklearn_djl_all', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_djl_all::Python=file:/opt/ml/model/sklearn_djl_all_formatters_v4.zip"
            )

            # Test DJL decorators
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_djl_all",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.777  # Custom predict formatter
            assert result["custom_sklearn"] == True  # Custom output formatter
            assert result["formatter_type"] == "djl_decorators"
            assert "model_type" in result

    def test_sklearn_djl_env_with_formatters(self):
        """Test sklearn DJL compatibility with env variables"""
        with Runner('cpu-full', 'sklearn_djl_env', download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=2",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=application/json"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_djl_env::Python=file:/opt/ml/model/sklearn_djl_all_formatters_v4.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_djl_env",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.777
            assert result["custom_sklearn"] == True
            assert result["formatter_type"] == "djl_decorators"

    def test_sklearn_djl_input_output_formatters(self):
        """Test sklearn DJL input_formatter and output_formatter decorators"""
        with Runner('cpu-full', 'sklearn_djl_input_output',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_djl_io::Python=file:/opt/ml/model/sklearn_djl_input_output_v3.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_djl_io",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "original_prediction" in result
            assert "doubled_prediction" in result
            assert "prediction_sum" in result
            assert result["djl_output_formatter_used"] == True
            assert result["formatter_type"] == "djl_decorators"
            # Verify doubled prediction is actually double the original
            original = result["original_prediction"][0]
            doubled = result["doubled_prediction"][0]
            assert doubled == original * 2

    def test_sklearn_djl_invalid_input_formatter(self):
        """Test sklearn DJL input_formatter that returns invalid format (should fail)"""
        with Runner('cpu-full', 'sklearn_djl_invalid_input',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_djl_invalid::Python=file:/opt/ml/model/sklearn_djl_invalid_input_v3.zip"
            )

            # Test should fail because input_formatter returns raw list instead of numpy array
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_djl_invalid",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424  # Failed dependency - input processing error

    def test_xgboost_djl_all_formatters(self):
        """Test XGBoost handler with all four DJL decorators"""
        with Runner('cpu-full', 'xgboost_djl_all', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_djl_all::Python=file:/opt/ml/model/xgboost_djl_all_formatters.zip"
            )

            # Test DJL decorators
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_djl_all",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.999  # Custom predict formatter
            assert result["custom_xgb"] == True  # Custom output formatter
            assert result["formatter_type"] == "djl_decorators"
            assert "model_type" in result

    def test_xgboost_djl_env_with_formatters(self):
        """Test XGBoost DJL compatibility with env variables"""
        with Runner('cpu-full', 'xgboost_djl_env', download=True) as r:
            env = [
                "SAGEMAKER_NUM_MODEL_WORKERS=2",
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT=application/json"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m xgboost_djl_env::Python=file:/opt/ml/model/xgboost_djl_all_formatters.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_djl_env",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == 0.999
            assert result["custom_xgb"] == True
            assert result["formatter_type"] == "djl_decorators"

    def test_xgboost_djl_input_output_formatters(self):
        """Test XGBoost DJL input_formatter and output_formatter decorators"""
        with Runner('cpu-full', 'xgboost_djl_input_output',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_djl_io::Python=file:/opt/ml/model/xgboost_djl_input_output_v3.zip"
            )

            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_djl_io",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "original_prediction" in result
            assert "doubled_prediction" in result
            assert "prediction_sum" in result
            assert result["djl_output_formatter_used"] == True
            assert result["formatter_type"] == "djl_decorators"
            # Verify doubled prediction is actually double the original
            original = result["original_prediction"][0]
            doubled = result["doubled_prediction"][0]
            assert doubled == original * 2

    def test_xgboost_djl_invalid_input_formatter(self):
        """Test XGBoost DJL input_formatter that returns invalid format (should fail)"""
        with Runner('cpu-full', 'xgboost_djl_invalid_input',
                    download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_djl_invalid::Python=file:/opt/ml/model/xgboost_djl_invalid_input_v3.zip"
            )

            # Test should fail because input_formatter returns raw list instead of numpy array
            test_data = {
                "features":
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_djl_invalid",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424  # Failed dependency - input processing error
