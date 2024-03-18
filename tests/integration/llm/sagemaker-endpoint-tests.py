import os
import sagemaker
import boto3
import time
from sagemaker.djl_inference import DJLModel, HuggingFaceAccelerateModel, DeepSpeedModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.utils import unique_name_from_base
from argparse import ArgumentParser
import numpy as np


def boolean_arg(value):
    return str(value).lower() == "true"


parser = ArgumentParser(
    description=
    "This Script deploys a model with predefined configuration to a SageMaker Inference Endpoint"
)
parser.add_argument(
    "model_name",
    help="The predefined model and configuration to use for deployment")
parser.add_argument("test_case",
                    help="The test case to execute",
                    choices=["djl", "no_code", "djl_mme"])
parser.add_argument(
    "image_type",
    help="Whether to use release or nightly images for testing",
    choices=["nightly", "release", "candidate"])

parser.add_argument(
    "run_benchmark",
    help="Whether to run benchmark and upload the metrics to cloudwatch",
    type=boolean_arg)

ROLE = "arn:aws:iam::185921645874:role/AmazonSageMaker-ExeuctionRole-IntegrationTests"
DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_PAYLOAD = {"inputs": "Deep Learning is"}
DEFAULT_BUCKET = "sm-integration-tests-rubikon-usw2"
RELEASE_VERSION = "0.23.0"
REGION = "us-west-2"

SINGLE_MODEL_ENDPOINT_CONFIGS = {
    "stable-diffusion-2-1-base": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "model_kwargs": {
            "dtype": "fp16",
            "number_of_partitions": 1,
        },
        "payload": {
            "prompt": "A picture of a cowboy in space"
        },
        "deserializer": sagemaker.deserializers.BytesDeserializer()
    },
    "gpt2-xl": {
        "model_id": "gpt2-xl",
        "model_kwargs": {
            "dtype": "fp32",
            "number_of_partitions": 2,
        }
    },
    "opt-1-3-b": {
        "model_id": "s3://djl-llm-sm-endpoint-tests/opt-1.3b/",
        "model_kwargs": {
            "dtype": "fp32",
            "number_of_partitions": 1,
        },
        "cls_to_use": HuggingFaceAccelerateModel,
    },
    "gpt-j-6b": {
        "model_id": "s3://djl-llm-sm-endpoint-tests/gpt-j-6b/",
        "model_kwargs": {
            "dtype": "bf16",
            "tensor_parallel_degree": 2,
            "parallel_loading": True,
        },
        "cls_to_use": DeepSpeedModel,
    },
    "pythia-12b": {
        "model_id": "EleutherAI/pythia-12b",
        "model_kwargs": {
            "dtype": "fp16",
            "tensor_parallel_degree": 4,
            "parallel_loading": True,
        },
        "partition_s3_uri": "s3://djl-llm-sm-endpoint-tests/pythia-12b-4p/",
        "cls_to_use": DeepSpeedModel,
    }
}

MME_CONFIGS = {
    "deepspeed-mme": {
        "models": [{
            'name': 'gpt-neo-2-7b',
            "model_id": "EleutherAI/gpt-neo-2.7B",
            "model_kwargs": {
                "dtype": "fp16",
                "number_of_partitions": 1,
            }
        }, {
            'name': 'opt-1-3b',
            "model_id": "s3://djl-llm-sm-endpoint-tests/opt-1.3b/",
            "model_kwargs": {
                "dtype": "fp16",
                "number_of_partitions": 1,
            }
        }],
        "instance_type":
        "ml.g5.8xlarge",
        "framework":
        "deepspeed",
    }
}

ENGINE_TO_METRIC_CONFIG_ENGINE = {"python": "accelerate"}

NIGHTLY_IMAGES = {
    "python":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:deepspeed-nightly",
    "deepspeed":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:deepspeed-nightly"
}

CANDIDATE_IMAGES = {
    "python":
    f"125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:{RELEASE_VERSION}-deepspeed",
    "deepspeed":
    f"125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:{RELEASE_VERSION}-deepspeed"
}


def get_sagemaker_session(default_bucket=DEFAULT_BUCKET,
                          default_bucket_prefix=None):
    return sagemaker.session.Session(
        boto3.session.Session(),
        default_bucket=default_bucket,
        default_bucket_prefix=default_bucket_prefix)


def delete_s3_test_artifacts(sagemaker_session):
    bucket = sagemaker_session.default_bucket()
    prefix = sagemaker_session.default_bucket_prefix
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).objects.filter(Prefix=prefix).delete()


def get_name_for_resource(name):
    cleaned_name = ''.join(filter(str.isalnum, name))
    base_name = "sm-integration-test-{}".format(cleaned_name)
    return unique_name_from_base(base_name)


def _upload_metrics(data):
    cw = boto3.client('cloudwatch')
    cw.put_metric_data(Namespace='LLM',
                       MetricData=[{
                           'MetricName': f"{data['metric_name']}-throughput",
                           'Unit': 'Count/Second',
                           'Value': data['throughput']
                       }, {
                           'MetricName': f"{data['metric_name']}-avg",
                           'Unit': 'Milliseconds',
                           'Value': data['avg']
                       }, {
                           'MetricName': f"{data['metric_name']}-p50",
                           'Unit': 'Milliseconds',
                           'Value': data['p50']
                       }, {
                           'MetricName': f"{data['metric_name']}-p90",
                           'Unit': 'Milliseconds',
                           'Value': data['p90']
                       }, {
                           'MetricName': f"{data['metric_name']}-p99",
                           'Unit': 'Milliseconds',
                           'Value': data['p99']
                       }])
    print(
        f"Uploaded metrics data with metric prefix {data['metric_name']} to AWS CloudWatch"
    )


def _get_metric_name(name, model, engine, instance_type):
    engine_name = ENGINE_TO_METRIC_CONFIG_ENGINE.get(engine, engine)

    num_partitions = 1
    if hasattr(model, 'number_of_partitions') and model.number_of_partitions:
        num_partitions = model.number_of_partitions
    return f"{name}-{engine_name}-{num_partitions}p-{instance_type}"


def _run_benchmarks(predictor, payload_data, metric_name, target_model=None):
    for _ in range(3):
        predictor.predict(data=payload_data, target_model=target_model)

    latencies = []
    iterations = 25
    begin = time.time()

    for _ in range(iterations):
        start = time.time()
        predictor.predict(data=payload_data, target_model=target_model)
        latencies.append((time.time() - start) * 1000)

    elapsed = (time.time() - begin) * 1000

    benchmark_data = {
        'metric_name': metric_name,
        'throughput': iterations / elapsed * 1000,
        'avg': sum(latencies) / iterations,
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p99': np.percentile(latencies, 99)
    }

    _upload_metrics(benchmark_data)


def mme_test(name, image_type, run_benchmark):
    config = MME_CONFIGS.get(name)
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("mme-tests"))
    models = config.get("models")
    framework = config.get("framework")
    instance_type = config.get("instance_type", DEFAULT_INSTANCE_TYPE)
    created_models = []
    mme = None
    predictor = None
    try:
        for model_config in models:
            model = DJLModel(model_config.get("model_id"),
                             ROLE,
                             name=get_name_for_resource(
                                 model_config.get("model_id") + '-mme'),
                             sagemaker_session=session,
                             **model_config.get("model_kwargs"))
            model.create()
            created_models.append(model)

        if image_type == "nightly":
            mme_image_uri = NIGHTLY_IMAGES[framework]
        elif image_type == "candidate":
            mme_image_uri = CANDIDATE_IMAGES[framework]
        else:
            mme_image_uri = sagemaker.image_uris.retrieve(
                framework="djl-" + framework,
                version=RELEASE_VERSION,
                region=REGION)
        mme = MultiDataModel(get_name_for_resource(name),
                             "s3://" + session.default_bucket() + '/' +
                             session.default_bucket_prefix,
                             config.get("prefix"),
                             image_uri=mme_image_uri,
                             role=ROLE,
                             sagemaker_session=session,
                             predictor_cls=sagemaker.predictor.Predictor)

        predictor = mme.deploy(
            1,
            instance_type,
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer())
        for i, model in enumerate(list(mme.list_models())):
            outputs = predictor.predict(DEFAULT_PAYLOAD, target_model=model)
            print(outputs)

            if run_benchmark:
                _run_benchmarks(predictor=predictor,
                                payload_data=DEFAULT_PAYLOAD,
                                metric_name=_get_metric_name(
                                    models[i]['name'], created_models[i],
                                    framework, instance_type),
                                target_model=model)

    except Exception as e:
        print(f"Encountered error for creating model {name}. Exception: {e}")
        raise e
    finally:
        delete_s3_test_artifacts(session)
        for m in created_models:
            m.delete_model()
        if mme:
            mme.delete_model()
        if predictor:
            predictor.delete_endpoint()


def single_model_endpoint_test(name, image_type, run_benchmark):
    config = SINGLE_MODEL_ENDPOINT_CONFIGS.get(name)
    data = config.get("payload", DEFAULT_PAYLOAD)
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("single_endpoint-tests"))
    model = None
    predictor = None
    try:
        model_cls = config.get("cls_to_use", DJLModel)
        model = model_cls(
            config.get("model_id"),
            ROLE,
            sagemaker_session=session,
            name=get_name_for_resource(name),
            **config.get("model_kwargs"),
        )
        engine_name = model.engine.value[0].lower()
        if image_type == "nightly":
            model.image_uri = NIGHTLY_IMAGES[engine_name]
        elif image_type == "candidate":
            model.image_uri = CANDIDATE_IMAGES[engine_name]

        if config.get("partition", False):
            model.partition(instance_type=DEFAULT_INSTANCE_TYPE,
                            s3_output_uri=config.get("partition_s3_uri"))

        predictor = model.deploy(DEFAULT_INSTANCE_TYPE,
                                 endpoint_name=get_name_for_resource(name),
                                 serializer=config.get("serializer", None),
                                 deserializer=config.get("deserializer", None))
        outputs = predictor.predict(data=data)
        print(outputs)

        if run_benchmark:
            _run_benchmarks(predictor=predictor,
                            payload_data=data,
                            metric_name=_get_metric_name(
                                name, model, engine_name,
                                DEFAULT_INSTANCE_TYPE),
                            target_model=None)

    except Exception as e:
        print(f"Encountered error for creating model {name}. Exception: {e}")
        raise e
    finally:
        delete_s3_test_artifacts(session)
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    test_case = args.test_case
    image_type = args.image_type
    if test_case == "djl":
        single_model_endpoint_test(model_name, image_type, args.run_benchmark)
    elif test_case == "djl_mme":
        mme_test(model_name, image_type, args.run_benchmark)
    else:
        raise ValueError(
            f"{test_case} is not a valid test case. Valid choices: [djl, no_code, djl_mme])"
        )
