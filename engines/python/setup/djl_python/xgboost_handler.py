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
from djl_python.custom_formatter_handling import CustomFormatterHandler, CustomFormatterError
from djl_python.import_utils import xgboost as xgb


class XGBoostHandler(CustomFormatterHandler):

    def __init__(self):
        super().__init__()
        self.model = None
        self.initialized = False
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
        try:
            self.load_formatters(model_dir)
        except CustomFormatterError as e:
            raise

        init_result = self.apply_init_handler(model_dir)
        if init_result is not None:
            self.model = init_result
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
            dmatrix = xgb.DMatrix(X)
            predictions = self.model.predict(dmatrix)

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


service = XGBoostHandler()


def handle(inputs: Input) -> Optional[Output]:
    if not service.initialized:
        service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return service.inference(inputs)
