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

import numpy as np
import os
import pickle as pkl
from io import StringIO
from typing import Optional
from djl_python import Input, Output
from djl_python.encode_decode import decode
from djl_python.utils import find_model_file
from djl_python.service_loader import get_annotated_function
from djl_python.import_utils import xgboost as xgb


class XGBoostHandler:

    def __init__(self):
        self.model = None
        self.initialized = False

    def initialize(self, properties: dict):
        model_dir = properties.get("model_dir")
        model_format = properties.get("model_format", "json")

        format_extensions = {
            "json": ["json"],
            "ubj": ["ubj"],
            "pickle": ["pkl", "pickle"],
            "xgb": ["xgb", "model", "bst"]
        }

        extensions = format_extensions.get(model_format)
        if not extensions:
            raise ValueError(
                f"Unsupported model format: {model_format}. Supported formats: json, ubj, pickle, xgb"
            )

        model_file = find_model_file(model_dir, extensions)
        if not model_file:
            raise FileNotFoundError(
                f"No model file found with format '{model_format}' in {model_dir}"
            )

        if model_format in ["json", "ubj"]:
            self.model = xgb.Booster()
            self.model.load_model(model_file)
        else:  # unsafe formats: pickle, xgb
            if properties.get("trust_insecure_model_files",
                              "false").lower() != "true":
                raise ValueError(
                    "trust_insecure_model_files must be set to 'true' to use unsafe formats (only json/ubj are secure by default)"
                )
            if model_format == "pickle":
                with open(model_file, 'rb') as f:
                    self.model = pkl.load(f)
            else:  # xgb format
                self.model = xgb.Booster()
                self.model.load_model(model_file)

        self.custom_input_formatter = get_annotated_function(
            model_dir, "is_input_formatter")
        self.custom_output_formatter = get_annotated_function(
            model_dir, "is_output_formatter")
        self.custom_predict_formatter = get_annotated_function(
            model_dir, "is_predict_formatter")

        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        content_type = inputs.get_property("Content-Type")
        accept = inputs.get_property("Accept") or "application/json"

        # Validate accept type (skip validation if custom output formatter is provided)
        if not self.custom_output_formatter:
            supported_accept_types = ["application/json", "text/csv"]
            if not any(supported_type in accept
                       for supported_type in supported_accept_types):
                raise ValueError(
                    f"Unsupported Accept type: {accept}. Supported types: {supported_accept_types}"
                )

        # Input processing
        X = None
        if self.custom_input_formatter:
            X = self.custom_input_formatter(inputs)
        elif "text/csv" in content_type:
            X = decode(inputs, content_type, require_csv_headers=False)
        else:
            input_map = decode(inputs, content_type)
            data = input_map.get("inputs") if isinstance(input_map,
                                                         dict) else input_map
            X = np.array(data)

        if X is None or not hasattr(X, 'ndim'):
            raise ValueError(
                f"Input processing failed for content type {content_type}")

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.custom_predict_formatter:
            predictions = self.custom_predict_formatter(self.model, X)
        else:
            dmatrix = xgb.DMatrix(X)
            predictions = self.model.predict(dmatrix, validate_features=False)

        # Output processing
        if self.custom_output_formatter:
            return self.custom_output_formatter(predictions)

        # Supports CSV/JSON outputs by default
        outputs = Output()
        if "text/csv" in accept:
            csv_buffer = StringIO()
            np.savetxt(csv_buffer, predictions, fmt='%s', delimiter=',')
            outputs.add(csv_buffer.getvalue().rstrip())
            outputs.add_property("Content-Type", "text/csv")
        else:
            outputs.add_as_json({"predictions": predictions.tolist()})
        return outputs


service = XGBoostHandler()


def handle(inputs: Input) -> Optional[Output]:
    if not service.initialized:
        service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return service.inference(inputs)
