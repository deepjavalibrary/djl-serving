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

        extensions = ["pkl", "model", "xgb", "bst", "json", "ubj"]

        model_file = find_model_file(model_dir, extensions)
        if not model_file:
            raise FileNotFoundError(
                f"No XGBoost model file found in {model_dir}")

        try:
            with open(model_file, 'rb') as f:
                self.model = pkl.load(f)
        except Exception:
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
