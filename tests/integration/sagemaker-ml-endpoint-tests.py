#!/usr/bin/env python3

import sagemaker
import boto3
import json
from sagemaker import Model, Predictor
from sagemaker.utils import unique_name_from_base
from sagemaker.serializers import JSONSerializer, CSVSerializer
from sagemaker.deserializers import JSONDeserializer, CSVDeserializer
from sagemaker.multidatamodel import MultiDataModel
from argparse import ArgumentParser

ROLE = "arn:aws:iam::185921645874:role/AmazonSageMaker-ExeuctionRole-IntegrationTests"
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_BUCKET = "sm-integration-tests-rubikon-usw2"

# DJL Serving CPU images
CANDIDATE_IMAGES = {
    "cpu-full":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving-cpu-full-test:latest"
}

# Test configurations using S3 URIs
SKLEARN_CONFIGS = {
    "sklearn-sagemaker-formatters": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/sklearn_custom_model_sm_v2.tar",
        "payload": {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                         [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    },
    "sklearn-djl-formatters": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/sklearn_djl_all_formatters_v4.tar",
        "payload": {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                         [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    },
    "sklearn-skops-basic": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/sklearn_skops_model.tar",
        "payload": {
            "inputs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                       [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "OPTION_SKOPS_TRUSTED_TYPES":
            "sklearn.ensemble._forest.RandomForestClassifier",
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    }
}

XGBOOST_CONFIGS = {
    "xgboost-sagemaker-formatters": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/xgboost_sagemaker_all.tar",
        "payload": {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                         [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    },
    "xgboost-djl-formatters": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/xgboost_djl_all_formatters.tar",
        "payload": {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                         [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    },
    "xgboost-basic": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/xgboost_model.tar",
        "payload": {
            "inputs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        "batch_payload": {
            "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                       [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        },
        "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
        "csv_batch_payload":
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0\n2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0\n3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0",
        "env_vars": {
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    }
}

# Multi-model endpoint configuration
MULTI_MODEL_CONFIGS = {
    "sklearn-multi": {
        "model_data":
        "s3://djl-llm-sm-endpoint-tests/skl_xgb/sklearn_multi_model_v2/",
        "models": {
            "model1": {
                "payload": {
                    "inputs":
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                },
                "csv_payload": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
            },
            "model2": {
                "payload": {
                    "inputs":
                    [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
                },
                "csv_payload": "2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0"
            }
        },
        "env_vars": {
            "OPTION_SKOPS_TRUSTED_TYPES":
            "sklearn.ensemble._forest.RandomForestClassifier",
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json",
            "SAGEMAKER_NUM_MODEL_WORKERS": "2"
        }
    }
}


def parse_args():
    parser = ArgumentParser(
        description="Deploy sklearn/xgboost models to SageMaker endpoints")
    parser.add_argument("model_name", help="Model configuration to use")
    parser.add_argument(
        "image_type",
        help="Image type (nightly, candidate, or full ECR URI)",
        nargs='?',
        default="nightly")
    parser.add_argument("-f",
                        "--framework",
                        help="Framework to test",
                        choices=["sklearn", "xgboost"])

    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--test-json",
                              help="Test JSON input/output only",
                              action="store_true")
    format_group.add_argument("--test-csv",
                              help="Test CSV input/output only",
                              action="store_true")
    format_group.add_argument("--test-both",
                              help="Test both JSON and CSV",
                              action="store_true")

    parser.add_argument("--test-batch",
                        help="Test batch predictions",
                        action="store_true")
    parser.add_argument("--test-multi-model",
                        help="Test multi-model endpoint",
                        action="store_true")

    return parser.parse_args()


def get_image_uri(image_type):
    if image_type == 'nightly':
        return NIGHTLY_IMAGES["cpu-full"]
    elif image_type == 'candidate':
        return CANDIDATE_IMAGES["cpu-full"]
    elif '.dkr.ecr.' in image_type or image_type.startswith(
            'deepjavalibrary/'):
        return image_type
    else:
        raise ValueError(
            f"Unknown image type: {image_type}. Use 'nightly', 'candidate', or full ECR URI"
        )


def get_sagemaker_session(default_bucket=DEFAULT_BUCKET,
                          default_bucket_prefix=None):
    return sagemaker.session.Session(
        boto3.session.Session(region_name='us-west-2'),
        default_bucket=default_bucket,
        default_bucket_prefix=default_bucket_prefix)


def get_name_for_resource(name):
    cleaned_name = ''.join(filter(str.isalnum, name))
    base_name = f"sm-ml-test-{cleaned_name}"
    return unique_name_from_base(base_name)


def test_endpoint(framework,
                  model_name,
                  image_type,
                  test_format="json",
                  test_batch=False):
    """Test sklearn/xgboost model on SageMaker endpoint"""
    configs = SKLEARN_CONFIGS if framework == "sklearn" else XGBOOST_CONFIGS
    config = configs[model_name]

    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource(
            f"{framework}-{model_name}"))

    predictor = None
    model = None

    try:
        # Add SageMaker environment variables if specified
        env_vars = config.get("env_vars", {})

        model = Model(image_uri=get_image_uri(image_type),
                      model_data=config["model_data"],
                      role=ROLE,
                      name=get_name_for_resource(f"{framework}-{model_name}"),
                      env=env_vars,
                      sagemaker_session=session)

        print(f"Deploying {framework} endpoint for {model_name}...")
        if env_vars:
            print(f"Using environment variables: {env_vars}")

        endpoint_name = get_name_for_resource(
            f"{framework}-{model_name}-endpoint")
        model.deploy(
            initial_instance_count=1,
            instance_type=DEFAULT_INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )

        predictor = Predictor(endpoint_name=endpoint_name,
                              sagemaker_session=session,
                              serializer=JSONSerializer(),
                              deserializer=JSONDeserializer())

        if test_format in ["json", "both"]:
            print("Testing JSON prediction...")
            result = predictor.predict(config["payload"])
            print(f"JSON Result: {result}")

            if test_batch and "batch_payload" in config:
                print("Testing JSON batch prediction...")
                batch_result = predictor.predict(config["batch_payload"])
                print(f"JSON Batch Result: {batch_result}")

        if test_format in ["csv", "both"] and "csv_payload" in config:
            print("Testing CSV prediction...")
            predictor.serializer = CSVSerializer()
            predictor.deserializer = CSVDeserializer()
            csv_result = predictor.predict(config["csv_payload"])
            print(f"CSV Result: {csv_result}")

            if test_batch and "csv_batch_payload" in config:
                print("Testing CSV batch prediction...")
                csv_batch_result = predictor.predict(
                    config["csv_batch_payload"])
                print(f"CSV Batch Result: {csv_batch_result}")

        batch_msg = " with batch" if test_batch else ""
        print(
            f"✓ Successfully tested {framework} model: {model_name}{batch_msg}"
        )

    except Exception as e:
        print(f"✗ Error testing {framework} model {model_name}: {e}")
        raise e
    finally:
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


def test_multi_model_endpoint(model_name, image_type, test_format="json"):
    """Test multi-model endpoint"""
    config = MULTI_MODEL_CONFIGS[model_name]

    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource(f"multi-{model_name}"))

    predictor = None
    model = None

    try:
        env_vars = config.get("env_vars", {})

        # Use MultiDataModel for multi-model endpoints
        model_s3_folder = config["model_data"].replace(".tar", "/")
        model = MultiDataModel(
            name=get_name_for_resource(f"multi-{model_name}"),
            model_data_prefix=model_s3_folder,
            image_uri=get_image_uri(image_type),
            role=ROLE,
            env=env_vars,
            sagemaker_session=session)

        print(f"Deploying multi-model endpoint for {model_name}...")
        if env_vars:
            print(f"Using environment variables: {env_vars}")

        endpoint_name = get_name_for_resource(f"multi-{model_name}-endpoint")
        model.deploy(
            initial_instance_count=1,
            instance_type=DEFAULT_INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )

        predictor = Predictor(endpoint_name=endpoint_name,
                              sagemaker_session=session,
                              serializer=JSONSerializer(),
                              deserializer=JSONDeserializer())

        # Test each model in the multi-model endpoint
        for model_id, model_config in config["models"].items():
            print(f"Testing model: {model_id}")

            if test_format in ["json", "both"]:
                print(f"Testing JSON prediction for {model_id}...")
                # Add model target header for multi-model endpoint
                result = predictor.predict(
                    model_config["payload"],
                    initial_args={"TargetModel": f"{model_id}.tar"})
                print(f"JSON Result for {model_id}: {result}")

            if test_format in ["csv", "both"
                               ] and "csv_payload" in model_config:
                print(f"Testing CSV prediction for {model_id}...")
                predictor.serializer = CSVSerializer()
                predictor.deserializer = CSVDeserializer()
                csv_result = predictor.predict(
                    model_config["csv_payload"],
                    initial_args={"TargetModel": f"{model_id}.tar"})
                print(f"CSV Result for {model_id}: {csv_result}")
                # Reset to JSON for next model
                predictor.serializer = JSONSerializer()
                predictor.deserializer = JSONDeserializer()

        print(f"✓ Successfully tested multi-model endpoint: {model_name}")

    except Exception as e:
        print(f"✗ Error testing multi-model endpoint {model_name}: {e}")
        raise e
    finally:
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


if __name__ == "__main__":
    args = parse_args()

    # Determine test format
    if args.test_csv:
        test_format = "csv"
    elif args.test_both:
        test_format = "both"
    else:
        test_format = "json"  # Default

    # Handle multi-model endpoint testing
    if args.test_multi_model:
        if args.model_name not in MULTI_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown multi-model config: {args.model_name}. Available: {list(MULTI_MODEL_CONFIGS.keys())}"
            )
        test_multi_model_endpoint(args.model_name, args.image_type,
                                  test_format)
    else:
        # Auto-detect framework if not specified
        if args.framework:
            configs = SKLEARN_CONFIGS if args.framework == "sklearn" else XGBOOST_CONFIGS
            if args.model_name not in configs:
                raise ValueError(
                    f"Unknown {args.framework} model: {args.model_name}. Available: {list(configs.keys())}"
                )
            test_endpoint(args.framework, args.model_name, args.image_type,
                          test_format, args.test_batch)
        else:
            if args.model_name in SKLEARN_CONFIGS:
                test_endpoint("sklearn", args.model_name, args.image_type,
                              test_format, args.test_batch)
            elif args.model_name in XGBOOST_CONFIGS:
                test_endpoint("xgboost", args.model_name, args.image_type,
                              test_format, args.test_batch)
            else:
                raise ValueError(
                    f"Unknown model: {args.model_name}. Available sklearn: {list(SKLEARN_CONFIGS.keys())}, xgboost: {list(XGBOOST_CONFIGS.keys())}, multi-model: {list(MULTI_MODEL_CONFIGS.keys())}"
                )
