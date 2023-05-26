import sagemaker
import boto3
from sagemaker.djl_inference import DJLModel, HuggingFaceAccelerateModel, DeepSpeedModel, FasterTransformerModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.utils import unique_name_from_base
from argparse import ArgumentParser

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

ROLE = "arn:aws:iam::185921645874:role/AmazonSageMaker-ExeuctionRole-IntegrationTests"
DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_PAYLOAD = {"inputs": "Deep Learning is"}
DEFAULT_BUCKET = "sm-integration-tests-rubikon"

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
        "model_id": "s3://djl-llm/opt-1.3b/",
        "model_kwargs": {
            "dtype": "fp32",
            "number_of_partitions": 1,
        },
        "partition": True,
        "cls_to_use": HuggingFaceAccelerateModel,
    },
    "flan-t5-xxl": {
        "model_id": "s3://djl-llm/flan-t5-xxl/",
        "model_kwargs": {
            "dtype": "fp32",
            "tensor_parallel_degree": 4,
        },
        "partition": True,
        "cls_to_use": FasterTransformerModel,
    },
    "gpt-j-6b": {
        "model_id": "s3://djl-llm/gpt-j-6b/",
        "model_kwargs": {
            "dtype": "bf16",
            "tensor_parallel_degree": 2,
            "parallel_loading": True,
        },
        "partition": True,
        "cls_to_use": DeepSpeedModel,
    },
    "pythia-12b": {
        "model_id": "EleutherAI/pythia-12b",
        "model_kwargs": {
            "dtype": "fp16",
            "tensor_parallel_degree": 4,
            "parallel_loading": True,
        },
        "partition": True,
        "partition_s3_uri": "s3://djl-llm/pythia-12b-4p/",
        "cls_to_use": DeepSpeedModel,
    }
}

HUGGING_FACE_NO_CODE_CONFIGS = {
    "gpt-neo-2-7-b": {
        "env": {
            "HF_MODEL_ID": "EleutherAI/gpt-neo-2.7B",
            "TENSOR_PARALLEL_DEGREE": "1",
        },
        "image_uri":
        sagemaker.image_uris.retrieve(framework="djl-deepspeed",
                                      version="0.22.1",
                                      region="us-east-1")
    },
    "bloom-7b1": {
        "env": {
            "HF_MODEL_ID": "bigscience/bloom-7b1"
        },
        "image_uri":
        sagemaker.image_uris.retrieve(framework="djl-fastertransformer",
                                      version="0.22.1",
                                      region="us-east-1")
    }
}

MME_CONFIGS = {
    "deepspeed-mme": {
        "models": [{
            "model_id": "EleutherAI/gpt-neo-2.7B",
            "model_kwargs": {
                "dtype": "fp16",
                "number_of_partitions": 1,
            }
        }, {
            "model_id": "s3://djl-llm/opt-1.3b/",
            "model_kwargs": {
                "dtype": "fp16",
                "number_of_partitions": 1,
            }
        }],
        "instance_type":
        "ml.g5.8xlarge",
        "image_uri":
        "125045733377.dkr.ecr.us-east-1.amazonaws.com/djl-serving:0.22.1-deepspeed",
    }
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


def mme_test(name):
    config = MME_CONFIGS.get(name)
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("mme-tests"))
    models = config.get("models")
    created_models = []
    mme = None
    predictor = None
    try:
        for model_config in models:
            model = DJLModel(model_config.get("model_id"),
                             ROLE,
                             image_uri=config.get("image_uri"),
                             name=get_name_for_resource(
                                 model_config.get("model_id") + '-mme'),
                             sagemaker_session=session,
                             **model_config.get("model_kwargs"))
            model.create()
            created_models.append(model)

        mme = MultiDataModel(get_name_for_resource(name),
                             "s3://" + session.default_bucket() + '/' +
                             session.default_bucket_prefix,
                             config.get("prefix"),
                             image_uri=config.get("image_uri"),
                             role=ROLE,
                             sagemaker_session=session,
                             predictor_cls=sagemaker.predictor.Predictor)

        predictor = mme.deploy(
            1,
            config.get("instance_type", DEFAULT_INSTANCE_TYPE),
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer())
        for model in list(mme.list_models()):
            outputs = predictor.predict(DEFAULT_PAYLOAD, target_model=model)
            print(outputs)

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


def no_code_endpoint_test(name):
    config = HUGGING_FACE_NO_CODE_CONFIGS.get(name)
    data = config.get("payload", DEFAULT_PAYLOAD)
    session = get_sagemaker_session(
        default_bucket_prefix=get_name_for_resource("no-code-tests"))
    model = None
    predictor = None
    try:
        model = HuggingFaceModel(
            role=ROLE,
            env=config.get("env"),
            sagemaker_session=session,
            image_uri=config.get("image_uri"),
            name=get_name_for_resource(name),
        )

        predictor = model.deploy(instance_type=DEFAULT_INSTANCE_TYPE,
                                 initial_instance_count=1,
                                 endpoint_name=get_name_for_resource(name),
                                 serializer=config.get("serializer", None),
                                 deserializer=config.get("deserializer", None))
        outputs = predictor.predict(data=data)
        print(outputs)
    except Exception as e:
        print(f"Encountered error for creating model {name}. Exception: {e}")
        raise e
    finally:
        delete_s3_test_artifacts(session)
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


def single_model_endpoint_test(name):
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

        if config.get("partition", False):
            model.partition(instance_type=DEFAULT_INSTANCE_TYPE,
                            s3_output_uri=config.get("partition_s3_uri"))

        predictor = model.deploy(DEFAULT_INSTANCE_TYPE,
                                 endpoint_name=get_name_for_resource(name),
                                 serializer=config.get("serializer", None),
                                 deserializer=config.get("deserializer", None))
        outputs = predictor.predict(data=data)
        print(outputs)
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
    if test_case == "djl":
        single_model_endpoint_test(model_name)
    elif test_case == "no_code":
        no_code_endpoint_test(model_name)
    elif test_case == "djl_mme":
        mme_test(model_name)
    else:
        raise ValueError(
            f"{test_case} is not a valid test case. Valid choices: [djl, no_code, djl_mme])"
        )
