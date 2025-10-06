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
from djl_python.import_utils import joblib, cloudpickle, skops_io as sio


class SklearnHandler:

    def __init__(self):
        self.model = None
        self.initialized = False

    def _get_trusted_types(self):
        trusted_types_str = os.environ.get("SKLEARN_SKOPS_TRUSTED_TYPES", "")
        if not trusted_types_str:
            raise ValueError(
                "SKLEARN_SKOPS_TRUSTED_TYPES environment variable must be set to load skops models. "
                "Example: SKLEARN_SKOPS_TRUSTED_TYPES='sklearn.ensemble._forest.RandomForestClassifier,numpy.ndarray'"
            )
        return [t.strip() for t in trusted_types_str.split(",") if t.strip()]

    def initialize(self, properties: dict):
        model_dir = properties.get("model_dir")
        model_format = os.environ.get("SKLEARN_MODEL_FORMAT", "skops")

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

        model_file = find_model_file(model_dir, extensions)
        if not model_file:
            raise FileNotFoundError(
                f"No model file found with format '{model_format}' in {model_dir}"
            )

        if model_format == "skops":
            trusted_types = self._get_trusted_types()
            self.model = sio.load(model_file, trusted=trusted_types)
        else:
            if os.environ.get("TRUST_INSECURE_PICKLE_FILES",
                              "").lower() != "true":
                raise ValueError(
                    f"TRUST_INSECURE_PICKLE_FILES must be set to 'true' to use {model_format} format (only skops is secure by default)"
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
        accept = inputs.get_property("Accept") or "application/json"

        if "text/csv" in content_type:
            X = decode(inputs, content_type, require_csv_headers=False)
        else:
            input_map = decode(inputs, content_type)
            data = input_map.get("inputs") if isinstance(input_map,
                                                         dict) else input_map
            X = np.array(data)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = self.model.predict(X)

        outputs = Output()
        if "text/csv" in accept:
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
