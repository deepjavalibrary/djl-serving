import sagemaker
import boto3
import time
from sagemaker import Model, Predictor
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.utils import unique_name_from_base
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from argparse import ArgumentParser
import numpy as np

ROLE = "arn:aws:iam::185921645874:role/AmazonSageMaker-ExeuctionRole-IntegrationTests"
DEFAULT_SME_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_MME_INSTANCE_TYPE = "ml.g5.8xlarge"
DEFAULT_PAYLOAD = {"inputs": "Deep Learning is"}
DEFAULT_BUCKET = "sm-integration-tests-rubikon-usw2"
REGION = "us-west-2"

NIGHTLY_IMAGES = {
    "lmi":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:lmi-nightly",
    "tensorrt-llm":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:tensorrt-llm-nightly",
    "neuronx":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:pytorch-inf2-nightly",
}

CANDIDATE_IMAGES = {
    # TODO: update this to new tag once lmi rename has been applied to newer release
    "lmi":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:{version}-deepspeed",
    "tensorrt-llm":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:{version}-tensorrt-llm",
    "neuronx":
    "125045733377.dkr.ecr.us-west-2.amazonaws.com/djl-serving:{version}-pytorch-inf2"
}

SINGLE_MODEL_ENDPOINT_CONFIGS = {
    # This tests the uncompressed model artifact SM capability (network isolation use-case)
    "llama3-8b": {
        "model_name": "llama3-8b",
        "s3_location": "s3://djl-llm-sm-endpoint-tests/llama-3-8b-hf/",
    },
    # This tests the S5CMD s3 downloading at runtime
    "mistral-7b": {
        "model_name": "mistral-7b",
        "env": {
            "HF_MODEL_ID": "s3://djl-llm-sm-endpoint-tests/mistral-7b/",
        }
    },
    # This tests the hf hub downloading at runtime
    "phi-2": {
        "model_name": "phi2",
        "env": {
            "HF_MODEL_ID": "microsoft/phi-2"
        }
    }
}

MME_CONFIGS = {
    "mme_common": {
        "models": [{
            "model_name": "flan-t5-small",
            "env": {
                "HF_MODEL_ID": "google/flan-t5-small",
            }
        }, {
            "model_name": "gpt2",
            "env": {
                "HF_MODEL_ID": "openai-community/gpt2",
            }
        }],
    }
}


def parse_args():
    parser = ArgumentParser(
        description=
        "This Script deploys a model with predefined configuration to a SageMaker Inference Endpoint"
    )
    parser.add_argument(
        "model_name",
        help="The predefined model and configuration to use for deployment")
    parser.add_argument(
        "endpoint_type",
        help=
        "The test case to execute (single model endpoint, multi model endpoint)",
        choices=["sme", "mme"])
    parser.add_argument(
        "image_type",
        help=
        "Whether to use nightly image, or candidate release image from internal ecr repo"
    )
    parser.add_argument("container",
                        help="Which container to use",
                        choices=["lmi", "tensorrt-llm", "neuronx"])
    parser.add_argument(
        "run_benchmark",
        help="Whether to run benchmark and upload the metrics to cloudwatch",
        type=boolean_arg)
    return parser.parse_args()


def boolean_arg(value):
    return str(value).lower() == "true"


def get_image_uri(image_type, framework_tag):
    if image_type == 'nightly':
        return NIGHTLY_IMAGES[framework_tag]
    else:
        image_uri = CANDIDATE_IMAGES[framework_tag]
        return image_uri.format(version=image_type)


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
    num_partitions = 1
    if hasattr(model, 'number_of_partitions') and model.number_of_partitions:
        num_partitions = model.number_of_partitions
    return f"{name}-{engine}-{num_partitions}p-{instance_type}"


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


def mme_test(name, image_type, framework, run_benchmark):
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("mme-tests"))
    created_models = []
    mme = None
    predictor = None
    config = MME_CONFIGS.get(name)
    try:
        for model_config in config.get("models"):
            model = Model(image_uri=get_image_uri(image_type, framework),
                          role=ROLE,
                          env=model_config.get("env"),
                          name=get_name_for_resource(f"djl-{framework}-mme"))
            model.create()
            created_models.append(model)

        mme = MultiDataModel(
            get_name_for_resource(f"djl-{framework}-mme"),
            "s3://" + session.default_bucket() + '/' +
            session.default_bucket_prefix,
            image_uri=get_image_uri(image_type, framework),
            role=ROLE,
            sagemaker_session=session,
        )

        predictor = mme.deploy(1,
                               DEFAULT_MME_INSTANCE_TYPE,
                               serializer=JSONSerializer(),
                               deserializer=JSONDeserializer())
        for i, model in enumerate(list(mme.list_models())):
            outputs = predictor.predict(DEFAULT_PAYLOAD, target_model=model)
            print(outputs)

            if run_benchmark:
                _run_benchmarks(predictor=predictor,
                                payload_data=DEFAULT_PAYLOAD,
                                metric_name=_get_metric_name(
                                    models[i]['name'], created_models[i],
                                    framework, DEFAULT_MME_INSTANCE_TYPE),
                                target_model=model)

    except Exception as e:
        print(f"Encountered error for creating model mme. Exception: {e}")
        raise e
    finally:
        delete_s3_test_artifacts(session)
        for m in created_models:
            m.delete_model()
        if mme:
            mme.delete_model()
        if predictor:
            predictor.delete_endpoint()


def sme_test(name, image_type, framework, run_benchmark):
    config = SINGLE_MODEL_ENDPOINT_CONFIGS.get(name)
    data = config.get("payload", DEFAULT_PAYLOAD)
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("single_endpoint-tests"))
    model_name = config.get("model_name")
    env = config.get("env")
    instance_type = config.get("instance_type", DEFAULT_SME_INSTANCE_TYPE)
    s3_location = config.get("s3_location", None)
    image_uri = get_image_uri(image_type, framework)
    model_data = None
    if s3_location is not None:
        model_data = {
            'S3DataSource': {
                'S3Uri': s3_location,
                'S3DataType': 'S3Prefix',
                'CompressionType': 'None'
            }
        }

    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=ROLE,
        env=env,
        name=get_name_for_resource(model_name),
    )
    endpoint_name = get_name_for_resource(model_name + "-endpoint")
    predictor = None
    try:
        model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )

        if run_benchmark:
            _run_benchmarks(predictor=predictor,
                            payload_data=data,
                            metric_name=_get_metric_name(
                                name, model, framework, instance_type),
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
    args = parse_args()
    model_name = args.model_name
    endpoint_type = args.endpoint_type
    image_type = args.image_type
    container = args.container
    run_benchmark = args.run_benchmark
    if endpoint_type == "sme":
        sme_test(model_name, image_type, container, run_benchmark)
    elif endpoint_type == "mme":
        mme_test(model_name, image_type, container, run_benchmark)
    else:
        raise ValueError(
            f"{endpoint_type} is not a valid endpoint type. Valid choices: [sme, mme])"
        )
