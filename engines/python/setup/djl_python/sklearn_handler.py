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
import glob
from typing import Optional
from djl_python import Input, Output
from djl_python.encode_decode import decode, decode_csv_numeric, encode_csv

try:
    import joblib
except ImportError:
    joblib = None

try:
    import skops.io as sio
except ImportError:
    sio = None

try:
    import cloudpickle
except ImportError:
    cloudpickle = None


class SklearnHandler:

    def __init__(self):
        self.model = None
        self.initialized = False

    def _find_model_file(self, model_dir: str, extensions: list) -> str:
        """Find model file with given extensions in model directory"""
        for ext in extensions:
            pattern = os.path.join(model_dir, f"*.{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]  # Return first match
        return None

    def initialize(self, properties: dict):
        model_dir = properties.get("model_dir")

        # Check for preferred model format from environment variable
        preferred_format = os.environ.get("SKLEARN_MODEL_FORMAT", "auto")

        if preferred_format != "auto":
            # Load specific format if requested
            model_file = self._find_model_file(model_dir, [preferred_format])
            if not model_file:
                raise FileNotFoundError(
                    f"No model file found with format '{preferred_format}' in {model_dir}"
                )

            if preferred_format == "skops" and sio:
                self.model = sio.load(
                    model_file,
                    trusted=sio.get_untrusted_types(file=model_file))
            elif preferred_format == "joblib" and joblib:
                self.model = joblib.load(model_file)
            elif preferred_format == "pkl":
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
            elif preferred_format == "cloudpkl" and cloudpickle:
                with open(model_file, 'rb') as f:
                    self.model = cloudpickle.load(f)
            else:
                raise ImportError(
                    f"Required library for format '{preferred_format}' not available"
                )
        else:
            # Auto-detect format in priority order
            model_file = None

            # Try skops first (most secure)
            if sio:
                model_file = self._find_model_file(model_dir, ["skops"])
                if model_file:
                    self.model = sio.load(
                        model_file,
                        trusted=sio.get_untrusted_types(file=model_file))

            # Try joblib
            if not model_file and joblib:
                model_file = self._find_model_file(model_dir, ["joblib"])
                if model_file:
                    self.model = joblib.load(model_file)

            # Try pickle
            if not model_file:
                model_file = self._find_model_file(model_dir,
                                                   ["pkl", "pickle"])
                if model_file:
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)

            # Try cloudpickle
            if not model_file and cloudpickle:
                model_file = self._find_model_file(model_dir, ["cloudpkl"])
                if model_file:
                    with open(model_file, 'rb') as f:
                        self.model = cloudpickle.load(f)

            if not model_file:
                raise FileNotFoundError(
                    f"No supported model file found in {model_dir}. Expected files with extensions: .skops, .joblib, .pkl, .pickle, .cloudpkl"
                )

        self.initialized = True

    def inference(self, inputs: Input) -> Output:
        content_type = inputs.get_property("Content-Type")
        accept = inputs.get_property("Accept") or "application/json"

        if "text/csv" in content_type:
            X = decode_csv_numeric(inputs)
        else:
            input_map = decode(inputs, content_type)
            data = input_map.get("inputs") if isinstance(input_map,
                                                         dict) else input_map
            X = np.array(data)

        predictions = self.model.predict(X)

        outputs = Output()
        if "text/csv" in accept:
            csv_data = "\n".join(str(pred) for pred in predictions)
            outputs.add(csv_data)
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
