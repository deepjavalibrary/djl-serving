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

from djl_python import Input, Output


def handle(inputs: Input) -> None:
    if inputs.is_empty():
        return None
    content_type = inputs.get_property("content-type")
    if content_type == "tensor/ndlist":
        return Output().add_as_numpy(np_list=inputs.get_as_numpy())
    elif content_type == "tensor/npz":
        return Output().add_as_npz(np_list=inputs.get_as_npz())
    elif content_type == "application/json":
        return Output().add_as_json(inputs.get_as_json())
    elif content_type is not None and content_type.startswith("text/"):
        return Output().add(inputs.get_as_string())
    else:
        return Output().add(inputs.get_as_bytes())
