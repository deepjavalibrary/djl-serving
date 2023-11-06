import yaml
import io
import logging
from pathlib import Path
from djl_python import Input
from djl_python import Output


class SageMakerInferenceService(object):

    def __init__(self) -> None:
        self.inference_spec = None
        self.model_dir = None
        self.model = None
        self.schema_builder = None
        self.inferenceSpec = None
        self.metadata = None
        self.default_serializer = None
        self.default_deserializer = None
        self.initialized = False

    def load_yaml(self, path: str):
        with open(path, mode="r") as file:
            return yaml.full_load(file)

    def load_metadata(self):
        metadata_path = Path(self.model_dir).joinpath('metadata.yaml')
        return self.load_yaml(metadata_path)

    def load_and_validate_pkl(self, path, hash_tag):
        import os
        import hmac
        import hashlib
        import cloudpickle
        with open(path, mode="rb") as file:
            buffer = file.read()
            secret_key = os.getenv('SAGEMAKER_SERVE_SECRET_KEY')
            stored_hash_tag = hmac.new(secret_key.encode(),
                                       msg=buffer,
                                       digestmod=hashlib.sha256).hexdigest()
            if not hmac.compare_digest(stored_hash_tag, hash_tag):
                raise Exception("Object is not valid: " + path)

        with open(path, mode="rb") as file:
            return cloudpickle.load(file)

    def load_default_schema(self, path):
        schema = self.load_yaml(path=path)
        if "input_deserializer" in schema:
            deserializer_name = schema.get("input_deserializer")
            if deserializer_name == "BytesDeserializer":
                from sagemaker.deserializers import BytesDeserializer
                self.default_deserializer = BytesDeserializer()
            elif deserializer_name == "NumpyDeserializer":
                from sagemaker.deserializers import NumpyDeserializer
                self.default_deserializer = NumpyDeserializer()
            elif deserializer_name == "JSONDeserializer":
                from sagemaker.deserializers import JSONDeserializer
                self.default_deserializer = JSONDeserializer()
            elif deserializer_name == "PandasDeserializer":
                from sagemaker.deserializers import PandasDeserializer
                self.default_deserializer = PandasDeserializer()
            elif deserializer_name == "TorchTensorDeserializer":
                from sagemaker.deserializers import TorchTensorDeserializer
                self.default_deserializer = TorchTensorDeserializer()
            elif deserializer_name == "PickleDeserializer":
                from sagemaker.deserializers import PickleDeserializer
                self.default_deserializer = PickleDeserializer()
            elif deserializer_name == "StringDeserializer":
                from sagemaker.deserializers import StringDeserializer
                self.default_deserializer = StringDeserializer()

        if "output_serializer" in schema:
            serializer_name = schema.get("output_serializer")
            if serializer_name == "DataSerializer":
                from sagemaker.serializers import DataSerializer
                self.default_serializer = DataSerializer()
            elif serializer_name == "NumpySerializer":
                from sagemaker.serializers import NumpySerializer
                self.default_serializer = NumpySerializer()
            elif serializer_name == "JSONSerializer":
                from sagemaker.serializers import JSONSerializer
                self.default_serializer = JSONSerializer()
            elif serializer_name == "CSVSerializer":
                from sagemaker.serializers import CSVSerializer
                self.default_serializer = CSVSerializer()
            elif serializer_name == "TorchTensorSerializer":
                from sagemaker.serializers import TorchTensorSerializer
                self.default_serializer = TorchTensorSerializer()
            elif serializer_name == "PickleSerializer":
                from sagemaker.serializers import PickleSerializer
                self.default_serializer = PickleSerializer()
            elif serializer_name == "StringSerializer":
                from sagemaker.serializers import StringSerializer
                self.default_serializer = StringSerializer()

    def load_pytorch_default(self, model_path):
        import torch
        return torch.jit.load(model_path)

    def invoke_pytorch_default(self, input):
        return self.model(input)

    def laod_xgboost_default(self, model_path):
        import xgboost
        return xgboost.Booster.load_model(model_path)

    def invoke_xgboost_default(self, input):
        return self.model.predict(input)

    def load(self):
        self.metadata = self.load_metadata()
        # Load schema and inference spec
        if "Schema" in self.metadata:
            schema_file_name = self.metadata.get("Schema")
            if ".yaml" in schema_file_name:
                # no customized schema builder case
                schema_builder_path = Path(self.model_dir).joinpath(
                    self.metadata.get("Schema")).absolute()
                self.load_default_schema(schema_builder_path)
            else:
                # load and validate customized schema builder
                schema_builder_path = Path(self.model_dir).joinpath(
                    self.metadata.get("Schema")).absolute()
                self.schema_builder = self.load_and_validate_pkl(
                    schema_builder_path, self.metadata.get("SchemaHMAC"))
        if "InferenceSpec" in self.metadata:
            inference_spec_path = Path(self.model_dir).joinpath(
                self.metadata.get("InferenceSpec")).absolute()
            self.inference_spec = self.load_and_validate_pkl(
                inference_spec_path, self.metadata.get("InferenceSpecHMAC"))

        # Load model
        model_name = self.metadata.get("Model")
        if self.inference_spec:
            self.model = self.inference_spec.load(self.model_dir)
        elif self.metadata.get("ModelType") == "PyTorchModel":
            model_path = Path(self.model_dir).joinpath(model_name).absolute()
            self.model = self.load_pytorch_default(model_path)
        elif self.metadata.get("ModelType") == "XGBoostModel":
            model_path = Path(self.model_dir).joinpath(model_name).absolute()
            self.model = self.laod_xgboost_default(model_path)
        else:
            raise Exception(
                "SageMaker model format does not support model type: " +
                self.metadata.get("ModelType"))

    def initialize(self, properties):
        # This method will initialize SageMaker service
        # The essential part is loading model and inferenceSpec
        self.model_dir = properties.get("model_dir")
        self.load()
        self.initialized = True
        logging.info(
            "SageMaker saved format entry-point is applied, service is initilized"
        )

    def preprocess_djl(self, inputs: Input):
        content_type = inputs.get_property("content-type")
        logging.info(f"Content-type is: {content_type}")
        if self.schema_builder:
            logging.info("Customized input deserializer is applied")
            try:
                if hasattr(self.schema_builder, "custom_input_translator"):
                    return self.schema_builder.custom_input_translator.deserialize(
                        io.BytesIO(inputs.get_as_bytes()), content_type)
                else:
                    raise Exception(
                        "No custom input translator in cutomized schema builder."
                    )
            except Exception as e:
                raise Exception(
                    "Encountered error in deserialize_request.") from e
        elif self.default_deserializer:
            return self.default_deserializer.deserialize(
                io.BytesIO(inputs.get_as_bytes()), content_type)

    def postproces_djl(self, output):
        if self.schema_builder:
            logging.info("Customized output serializer is applied")
            try:
                if hasattr(self.schema_builder, "custom_output_translator"):
                    return self.schema_builder.custom_output_translator.serialize(
                        output)
                else:
                    raise Exception(
                        "No custom output translator in cutomized schema builder."
                    )
            except Exception as e:
                raise Exception(
                    "Encountered error in serialize_response.") from e
        elif self.default_serializer:
            return self.default_serializer.serialize(output)

    def inference(self, inputs: Input):
        processed_input = self.preprocess_djl(inputs=inputs)
        if self.inference_spec:
            output = self.inference_spec.invoke(processed_input, self.model)
        elif self.metadata.get("ModelType") == "PyTorchModel":
            output = self.invoke_pytorch_default()
        elif self.metadata.get("ModelType") == "XGBoostModel":
            output = self.invoke_xgboost_default()
        else:
            raise Exception(
                "SageMaker model format does not support model type: " +
                self.metadata.get("ModelType"))
        processed_output = self.postproces_djl(output=output)
        output_data = Output()
        return output_data.add(processed_output)


_service = SageMakerInferenceService()


def handle(inputs: Input) -> Output:
    if not _service.initialized:
        properties = inputs.get_properties()
        _service.initialize(properties)

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
