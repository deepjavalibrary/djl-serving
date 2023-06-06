#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import base64
import csv
import datetime
import json
from io import BytesIO, StringIO
from djl_python.inputs import Input
from djl_python.outputs import Output

import numpy as np


def decode_csv(inputs: Input):  # type: (str) -> np.array
    stream = StringIO(inputs.get_as_string())
    # detects if the incoming csv has headers
    if not any(header in string_like.splitlines()[0].lower()
               for header in ["question", "context", "inputs"]):
        raise ValueError(
            "You need to provide the correct CSV with Header columns to use it with the inference toolkit default handler.",
        )
    # reads csv as io
    request_list = list(csv.DictReader(stream))
    if "inputs" in request_list[0].keys():
        return {"inputs": [entry["inputs"] for entry in request_list]}
    else:
        return {"inputs": request_list}


def encode_csv(content):  # type: (str) -> np.array
    stream = StringIO()
    if not isinstance(content, list):
        content = list(content)

    column_header = content[0].keys()
    writer = csv.DictWriter(stream, column_header)

    writer.writeheader()
    writer.writerows(content)
    return stream.getvalue()


def decode(inputs: Input, content_type: str, key=None):
    if not content_type or "application/json" in content_type:
        return inputs.get_as_json(key=key)
    elif "text/csv" in content_type:
        return decode_csv(inputs)
    elif "text/plain" in content_type:
        return {"inputs": [inputs.get_as_string(key=key)]}
    if content_type.startswith("image/"):
        return {"inputs": inputs.get_as_image(key=key)}
    elif content_type.startswith("audio/"):
        return {"inputs": inputs.get_as_bytes(key=key)}
    elif "tensor/npz" in content_type:
        return {"inputs": inputs.get_as_npz(key=key)}
    elif content_type in {"tensor/ndlist", "application/x-npy"}:
        return {"inputs": inputs.get_as_numpy(key=key)}
    elif content_type == "application/x-www-form-urlencoded":
        return {"inputs": inputs.get_as_string(key=key)}
    else:
        # "application/octet-stream"
        return {"inputs": inputs.get_as_bytes(key=key)}


def encode(outputs: Output, prediction, content_type: str, key=None):
    if not content_type or "application/json" in content_type:
        outputs.add_as_json(prediction, key=key)
        outputs.add_property("Content-Type", "application/json")
    elif "text/csv" in content_type:
        outputs.add_as_string(encode_csv(prediction), key=key)
        outputs.add_property("Content-Type", "text/csv")
    elif "tensor/npz" in content_type:
        outputs.add_as_npz(prediction, key=key)
        outputs.add_property("Content-Type", "tensor/npz")
    else:
        outputs.add_as_numpy(prediction, key=key)
        outputs.add_property("Content-Type", "tensor/ndlist")
