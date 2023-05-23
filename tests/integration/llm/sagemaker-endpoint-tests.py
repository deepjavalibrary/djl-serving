import sagemaker
import boto3
from sagemaker.djl_inference import DJLModel, HuggingFaceAccelerateModel, DeepSpeedModel, FasterTransformerModel
from sagemaker.huggingface import HuggingFaceModel
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
        "cls_to_use": HuggingFaceAccelerateModel,
    },
    "flan-t5-xxl": {
        "model_id": "s3://djl-llm/flan-t5-xxl/",
        "model_kwargs": {
            "dtype": "fp32",
            "tensor_parallel_degree": 4,
        },
        "cls_to_use": FasterTransformerModel,
    },
    "gpt-j-6b": {
        "model_id": "s3://djl-llm/gpt-j-6b/",
        "model_kwargs": {
            "dtype": "bf16",
            "tensor_parallel_degree": 2,
            "parallel_loading": True,
        },
        "cls_to_use": DeepSpeedModel,
    },
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

ROLE = "arn:aws:iam::185921645874:role/AmazonSageMaker-ExeuctionRole-IntegrationTests"
DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_PAYLOAD = {"inputs": "Deep Learning is"}


def no_code_endpoint_test(name, sagemaker_session):
    base_name = "{}-sm-integration-test-no-code".format(name)
    config = HUGGING_FACE_NO_CODE_CONFIGS.get(name)
    data = config.get("payload", DEFAULT_PAYLOAD)
    model = None
    predictor = None
    try:
        model = HuggingFaceModel(
            role=ROLE,
            env=config.get("env"),
            sagemaker_session=sagemaker_session,
            image_uri=config.get("image_uri"),
            name=unique_name_from_base(base_name),
        )

        predictor = model.deploy(
            instance_type=DEFAULT_INSTANCE_TYPE,
            initial_instance_count=1,
            endpoint_name=unique_name_from_base(base_name),
            serializer=config.get("serializer", None),
            deserializer=config.get("deserializer", None))
        outputs = predictor.predict(data=data)
        print(outputs)
    except Exception as e:
        print(f"Encountered error for creating model {name}. Exception: {e}")
        raise e
    finally:
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


def single_model_endpoint_test(name, sagemaker_session):
    base_name = "{}-sm-integration-test-djl".format(name)
    config = SINGLE_MODEL_ENDPOINT_CONFIGS.get(name)
    data = config.get("payload", DEFAULT_PAYLOAD)
    model = None
    predictor = None
    try:
        model_cls = config.get("cls_to_use", DJLModel)
        model = model_cls(
            config.get("model_id"),
            ROLE,
            sagemaker_session=sagemaker_session,
            name=unique_name_from_base(base_name),
            **config.get("model_kwargs"),
        )

        if config.get("partition", False):
            model.partition()

        predictor = model.deploy(
            DEFAULT_INSTANCE_TYPE,
            endpoint_name=unique_name_from_base(base_name),
            serializer=config.get("serializer", None),
            deserializer=config.get("deserializer", None))
        outputs = predictor.predict(data=data)
        print(outputs)
    except Exception as e:
        print(f"Encountered error for creating model {name}. Exception: {e}")
        raise e
    finally:
        if predictor:
            predictor.delete_endpoint()
        if model:
            model.delete_model()


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    test_case = args.test_case
    sagemaker_session = sagemaker.session.Session(boto3.session.Session())
    if test_case == "djl":
        single_model_endpoint_test(model_name, sagemaker_session)
    elif test_case == "no_code":
        no_code_endpoint_test(model_name, sagemaker_session)
    elif test_case == "djl_mme":
        print("MME Testing not Supported yet")
        pass
    else:
        raise ValueError(
            f"{test_case} is not a valid test case. Valid choices: [djl, no_code, djl_mme])"
        )
