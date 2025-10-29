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
from djl_python.custom_formatter_handling import load_custom_code
from djl_python.import_utils import xgboost as xgb


class XGBoostHandler:

    def __init__(self):
        self.model = None
        self.initialized = False
        self.custom_code = None
        self.init_properties = None

    def initialize(self, properties: dict):
        # Store initialization properties for use during inference
        self.init_properties = properties.copy()
        model_dir = properties.get("model_dir")
        model_format = (properties.get("model_format")
                        or os.environ.get("MODEL_FORMAT") or "json")

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

        # Load custom code
        self.custom_code = load_custom_code(model_dir)

        if self.custom_code.handlers.init_handler:
            self.model = self.custom_code.handlers.init_handler(model_dir)
        else:
            model_file = find_model_file(model_dir, extensions)
            if not model_file:
                raise FileNotFoundError(
                    f"No model file found with format '{model_format}' in {model_dir}"
                )

            if model_format in ["json", "ubj"]:
                self.model = xgb.Booster()
                self.model.load_model(model_file)
            else:  # unsafe formats: pickle, xgb
                trust_insecure = (properties.get("trust_insecure_model_files")
                                  or
                                  os.environ.get("TRUST_INSECURE_MODEL_FILES")
                                  or "false")
                if trust_insecure.lower() != "true":
                    raise ValueError(
                        "option.trust_insecure_model_files must be set to 'true' to use unsafe formats (only json/ubj are secure by default)"
                    )
                if model_format == "pickle":
                    with open(model_file, 'rb') as f:
                        self.model = pkl.load(f)
                else:  # xgb format
                    self.model = xgb.Booster()
                    self.model.load_model(model_file)

        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        content_type = inputs.get_property("Content-Type")
        properties = inputs.get_properties()
        # Use initialization properties as fallback for missing request properties
        default_accept = self.init_properties.get("default_accept",
                                                  "application/json")

        accept = inputs.get_property("Accept")

        # Treat */* as no preference, use default
        if accept == "*/*":
            accept = default_accept

        # Validate accept type (skip validation if custom output formatter is provided)
        if not self.custom_code.formatters.output_formatter:
            supported_accept_types = ["application/json", "text/csv"]
            if not any(supported_type in accept
                       for supported_type in supported_accept_types):
                raise ValueError(
                    f"Unsupported Accept type: {accept}. Supported types: {supported_accept_types}"
                )

        # Input processing
        X = None
        if self.custom_code.formatters.input_formatter:
            if self.custom_code.is_sagemaker_script:
                X = self.custom_code.formatters.input_formatter(
                    inputs.get_as_bytes(), content_type)
            else:
                X = self.custom_code.formatters.input_formatter(inputs)
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
        if self.custom_code.handlers.prediction_handler:
            predictions = self.custom_code.handlers.prediction_handler(
                X, self.model)
        else:
            dmatrix = xgb.DMatrix(X)
            predictions = self.model.predict(dmatrix)

        # Output processing
        outputs = Output()
        if self.custom_code.formatters.output_formatter:
            if self.custom_code.is_sagemaker_script:
                data = self.custom_code.formatters.output_formatter(
                    predictions, accept)
                outputs.add_property("Content-Type", accept)
            else:
                data = self.custom_code.formatters.output_formatter(
                    predictions)
            outputs.add(data)

        elif "text/csv" in accept:
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
