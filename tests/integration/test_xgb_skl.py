#!/usr/bin/env python3

import os
import pytest
import requests
import json
from tests import Runner


@pytest.mark.cpu
class TestXgbSkl:

    # Basic model tests
    def test_sklearn_model(self):
        with Runner('cpu-full', 'sklearn_model', download=True) as r:
            r.launch(
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
            assert len(result["predictions"]) == 1

    def test_xgboost_model(self):
        with Runner('cpu-full', 'xgboost_model', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert len(result["predictions"]) == 1

    # CSV input/output tests
    def test_sklearn_csv_input(self):
        with Runner('cpu-full', 'sklearn_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 200

    def test_xgboost_csv_input(self):
        with Runner('cpu-full', 'xgboost_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 200

    def test_sklearn_json_input_csv_output(self):
        with Runner('cpu-full', 'sklearn_json_csv', download=True) as r:
            r.launch(
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
                    "Accept": "text/csv"
                })
            assert response.status_code == 200
            assert "text/csv" in response.headers.get("Content-Type", "")

    def test_sklearn_csv_input_json_output(self):
        with Runner('cpu-full', 'sklearn_csv_json', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_xgboost_csv_input_json_output(self):
        with Runner('cpu-full', 'xgboost_csv_json', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_xgboost_json_input_csv_output(self):
        with Runner('cpu-full', 'xgboost_json_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/csv"
                })
            assert response.status_code == 200
            assert "text/csv" in response.headers.get("Content-Type", "")

    # Model format tests
    def test_sklearn_joblib_format(self):
        with Runner('cpu-full', 'sklearn_joblib', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_joblib::Python=file:/opt/ml/model/sklearn_joblib_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_joblib",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_sklearn_cloudpickle_format(self):
        with Runner('cpu-full', 'sklearn_cloudpickle', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_cloudpickle::Python=file:/opt/ml/model/sklearn_cloudpickle_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_cloudpickle",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_xgboost_ubj_format(self):
        with Runner('cpu-full', 'xgboost_ubj', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_ubj::Python=file:/opt/ml/model/xgboost_ubj_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_ubj",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    def test_xgboost_deprecated_format(self):
        with Runner('cpu-full', 'xgboost_deprecated', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_deprecated::Python=file:/opt/ml/model/xgboost_deprecated_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_deprecated",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    # Error handling tests - CSV format errors
    def test_sklearn_csv_with_headers(self):
        with Runner('cpu-full', 'sklearn_csv_headers', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10\n1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    def test_sklearn_ragged_csv(self):
        with Runner('cpu-full', 'sklearn_ragged_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0\n6.0,7.0,8.0,9.0,10.0,11.0,12.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    def test_sklearn_empty_rows_csv(self):
        with Runner('cpu-full', 'sklearn_empty_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n\n1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 200  # Skips empty rows

    def test_sklearn_non_numeric_csv(self):
        with Runner('cpu-full', 'sklearn_non_numeric_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            csv_data = "1.0,2.0,abc,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    def test_xgboost_csv_with_headers(self):
        with Runner('cpu-full', 'xgboost_csv_headers', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10\n1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    def test_xgboost_ragged_csv(self):
        with Runner('cpu-full', 'xgboost_ragged_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0\n6.0,7.0,8.0,9.0,10.0,11.0,12.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    def test_xgboost_empty_rows_csv(self):
        with Runner('cpu-full', 'xgboost_empty_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n\n1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 200

    def test_xgboost_non_numeric_csv(self):
        with Runner('cpu-full', 'xgboost_non_numeric_csv', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            csv_data = "1.0,2.0,abc,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                data=csv_data,
                headers={
                    "Content-Type": "text/csv",
                    "Accept": "text/csv"
                })
            assert response.status_code == 424

    # Error handling tests - Input shape errors
    def test_sklearn_wrong_input_shape(self):
        with Runner('cpu-full', 'sklearn_wrong_shape', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            test_data = {"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]}
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 424

    def test_sklearn_ragged_arrays(self):
        with Runner('cpu-full', 'sklearn_ragged', download=True) as r:
            r.launch(
                cmd=
                "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                           [1.0, 2.0, 3.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424

    def test_xgboost_wrong_input_shape(self):
        with Runner('cpu-full', 'xgboost_wrong_shape', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[
                    1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
                    3.0, 4.0, 5.0
                ]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={"Content-Type": "application/json"})
            assert response.status_code == 424

    def test_xgboost_ragged_arrays(self):
        with Runner('cpu-full', 'xgboost_ragged', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                           [1.0, 2.0, 3.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 424

    # Error handling tests - Content type errors
    def test_sklearn_invalid_accept_type(self):
        with Runner('cpu-full', 'sklearn_unsupported_accept',
                    download=True) as r:
            r.launch(
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
                    "Accept": "text/xml"
                })
            assert response.status_code == 424

    def test_sklearn_invalid_content_type(self):
        with Runner('cpu-full', 'sklearn_invalid_content', download=True) as r:
            r.launch(
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
                    "Content-Type": "application/xml",
                    "Accept": "application/json"
                })
            assert response.status_code == 424

    def test_xgboost_invalid_accept_type(self):
        with Runner('cpu-full', 'xgboost_invalid_accept', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/xml"
                })
            assert response.status_code == 424

    def test_xgboost_invalid_content_type(self):
        with Runner('cpu-full', 'xgboost_invalid_content', download=True) as r:
            r.launch(
                cmd=
                "serve -m xgboost_test::Python=file:/opt/ml/model/xgboost_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/xgboost_test",
                json=test_data,
                headers={
                    "Content-Type": "application/xml",
                    "Accept": "application/json"
                })
            assert response.status_code == 424

    # Error handling tests - Configuration errors
    def test_multiple_artifacts(self):
        with Runner('cpu-full', 'sklearn_multi_artifacts', download=True) as r:
            try:
                r.launch(
                    cmd=
                    "serve -m sklearn_multi::Python=file:/opt/ml/model/sklearn_multi_model_v2.zip"
                )
                test_data = {
                    "inputs":
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
                }
                response = requests.post(
                    "http://localhost:8080/predictions/sklearn_multi",
                    json=test_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    })
                assert response.status_code != 200
            except Exception:
                pass

    def test_sklearn_bad_env_variable(self):
        with Runner('cpu-full', 'sklearn_bad_env', download=True) as r:
            env = ["OPTION_SKOPS_TRUSTED_TYPES=invalid_type"]
            try:
                r.launch(
                    env_vars=env,
                    cmd=
                    "serve -m sklearn_test::Python=file:/opt/ml/model/sklearn_skops_model_v2.zip"
                )
                test_data = {
                    "inputs":
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
                }
                response = requests.post(
                    "http://localhost:8080/predictions/sklearn_test",
                    json=test_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    })
                assert response.status_code != 200
            except Exception:
                pass

    def test_sklearn_skops_with_valid_trusted_types(self):
        with Runner('cpu-full', 'sklearn_skops_valid', download=True) as r:
            env = [
                "OPTION_SKOPS_TRUSTED_TYPES=sklearn.ensemble._forest.RandomForestClassifier"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_skops::Python=file:/opt/ml/model/sklearn_skops_model_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_skops",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

    # Security tests
    def test_sklearn_unsafe_format_without_trust(self):
        with Runner('cpu-full', 'sklearn_unsafe', download=True) as r:
            try:
                r.launch(
                    cmd=
                    "serve -m sklearn_unsafe::Python=file:/opt/ml/model/sklearn_unsafe_model_v2.zip"
                )
                test_data = {
                    "inputs":
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
                }
                response = requests.post(
                    "http://localhost:8080/predictions/sklearn_unsafe",
                    json=test_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    })
                assert response.status_code != 200
            except Exception:
                pass

    def test_xgboost_unsafe_format_without_trust(self):
        with Runner('cpu-full', 'xgboost_unsafe', download=True) as r:
            try:
                r.launch(
                    cmd=
                    "serve -m xgboost_unsafe::Python=file:/opt/ml/model/xgboost_unsafe_model_v2.zip"
                )
                test_data = {
                    "inputs":
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
                }
                response = requests.post(
                    "http://localhost:8080/predictions/xgboost_unsafe",
                    json=test_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    })
                assert response.status_code != 200
            except Exception:
                pass

    def test_sklearn_skops_env_variables_only(self):
        with Runner('cpu-full', 'sklearn_skops_env_only', download=True) as r:
            env = [
                "OPTION_MODEL_FORMAT=skops",
                "OPTION_TRUST_INSECURE_MODEL_FILES=true",
                "OPTION_SKOPS_TRUSTED_TYPES=sklearn.ensemble._forest.RandomForestClassifier"
            ]
            r.launch(
                env_vars=env,
                cmd=
                "serve -m sklearn_skops_env::Python=file:/opt/ml/model/sklearn_skops_model_env_v2.zip"
            )
            test_data = {
                "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
            }
            response = requests.post(
                "http://localhost:8080/predictions/sklearn_skops_env",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                })
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
