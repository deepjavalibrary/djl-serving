#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import pickle
import numpy as np
import os
from io import StringIO
from typing import Optional
from djl_python import Input, Output
from djl_python.encode_decode import decode
from djl_python.utils import find_model_file
from djl_python.custom_formatter_handling import CustomFormatterHandler, CustomFormatterError
from djl_python.import_utils import joblib, cloudpickle, skops_io as sio


class SklearnHandler(CustomFormatterHandler):

    def __init__(self):
        super().__init__()
        self.model = None
        self.initialized = False
        self.init_properties = None

    def _get_trusted_types(self, properties: dict):
        trusted_types_str = properties.get("skops_trusted_types", "")
        if not trusted_types_str:
            raise ValueError(
                "option.skops_trusted_types must be set to load skops models. "
                "Example: option.skops_trusted_types='sklearn.ensemble._forest.RandomForestClassifier,numpy.ndarray'"
            )
        trusted_types = [
            t.strip() for t in trusted_types_str.split(",") if t.strip()
        ]
        print(f"Using trusted types for skops model loading: {trusted_types}")
        return trusted_types

    def initialize(self, properties: dict):
        # Store initialization properties for use during inference
        self.init_properties = properties.copy()
        model_dir = properties.get("model_dir")
        model_format = properties.get("model_format", "skops")

        format_extensions = {
            "skops": ["skops"],
            "joblib": ["joblib", "jl"],
            "pickle": ["pkl", "pickle"],
            "cloudpickle": ["pkl", "pickle", "cloudpkl"]
        }

        extensions = format_extensions.get(model_format)
        if not extensions:
            raise ValueError(
                f"Unsupported model format: {model_format}. Supported formats: skops, joblib, pickle, cloudpickle"
            )

        # Load custom code
        try:
            self.load_formatters(model_dir)
        except CustomFormatterError as e:
            raise

        # Load model
        init_result = self.apply_init_handler(model_dir)
        if init_result is not None:
            self.model = init_result
        else:
            model_file = find_model_file(model_dir, extensions)
            if not model_file:
                raise FileNotFoundError(
                    f"No model file found with format '{model_format}' in {model_dir}"
                )

            if model_format == "skops":
                trusted_types = self._get_trusted_types(properties)
                self.model = sio.load(model_file, trusted=trusted_types)
            else:
                if properties.get("trust_insecure_model_files",
                                  "false").lower() != "true":
                    raise ValueError(
                        f"option.trust_insecure_model_files must be set to 'true' to use {model_format} format (only skops is secure by default)"
                    )

                if model_format == "joblib":
                    self.model = joblib.load(model_file)
                elif model_format == "pickle":
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                elif model_format == "cloudpickle":
                    with open(model_file, 'rb') as f:
                        self.model = cloudpickle.load(f)

        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        content_type = inputs.get_property("Content-Type")
        properties = inputs.get_properties()
        default_accept = self.init_properties.get("default_accept",
                                                  "application/json")

        accept = inputs.get_property("Accept")

        # If no accept type is specified in the request, use default
        if accept == "*/*":
            accept = default_accept

        # Validate accept type (skip validation if custom output formatter is provided)
        if self.output_formatter is None:  # No formatter available
            if not any(supported_type in accept
                       for supported_type in ["application/json", "text/csv"]):
                raise ValueError(
                    f"Unsupported Accept type: {accept}. Supported types: application/json, text/csv"
                )

        # Input processing
        X = self.apply_input_formatter(inputs, content_type=content_type)
        if X is inputs:  # No formatter applied
            if "text/csv" in content_type:
                X = decode(inputs, content_type, require_csv_headers=False)
            else:
                input_map = decode(inputs, content_type)
                data = input_map.get("inputs") if isinstance(
                    input_map, dict) else input_map
                X = np.array(data)

        if X is None or not hasattr(X, 'ndim'):
            raise ValueError(
                f"Input processing failed for content type {content_type}")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = self.apply_prediction_handler(X, self.model)
        if predictions is None:
            predictions = self.model.predict(X)

        # Output processing
        outputs = Output()
        formatted_output = self.apply_output_formatter(predictions,
                                                       accept=accept)
        if formatted_output is not predictions:  # Formatter was applied
            if self.is_sagemaker_script:
                outputs.add_property("Content-Type", accept)
            outputs.add(formatted_output)
        elif "text/csv" in accept:
            csv_buffer = StringIO()
            np.savetxt(csv_buffer, predictions, fmt='%s', delimiter=',')
            outputs.add(csv_buffer.getvalue().rstrip())
            outputs.add_property("Content-Type", "text/csv")
        else:
            outputs.add_as_json({"predictions": predictions.tolist()})
        return outputs


service = SklearnHandler()


def handle(inputs: Input) -> Optional[Output]:
    if not service.initialized:
        service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return service.inference(inputs)
