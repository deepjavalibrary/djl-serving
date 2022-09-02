#!/usr/bin/env python
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import io

import torch

from djl_python.outputs import Output
from .service_loader import ModelService
from ts.service import Service


class TorchServeService(ModelService):

    def __init__(self, service, model_dir):
        super().__init__(None, model_dir)
        self.service = service

    def invoke_handler(self, function_name, inputs):
        if inputs.is_empty():
            # TS model has been initialized already, ignore init request
            return Output()

        request = dict()
        request["requestId"] = "0".encode("utf-8")
        request["headers"] = []
        request["parameters"] = []

        for k, v in inputs.get_properties().items():
            header = dict()
            header["name"] = k.encode("utf-8")
            header["value"] = v.encode("utf-8")
            request["headers"].append(header)

        content = inputs.get_content()
        for i in range(content.size()):
            k = content.key_at(i)
            if k is None:
                k = "data" if content.size() == 1 else ""
            else:
                k = k.lower()
            v = content.value_at(i)
            model_input = dict()
            model_input["name"] = k
            if k == "data":
                model_input["contentType"] = inputs.get_property(
                    "content-type")
            else:
                model_input["contentType"] = None
            model_input["value"] = v

            request["parameters"].append(model_input)

        ctx = self.service.context
        headers, input_batch, req_id_map = Service.retrieve_data_for_inference(
            [request])
        ctx.request_ids = req_id_map
        ctx.request_processor = headers
        ts_out = self.service._entry_point(input_batch, ctx)

        content_type = ctx.get_response_content_type(0)
        code, msg = ctx.get_response_status(0)
        code = 200 if code is None else code
        msg = "OK" if msg is None else msg
        outputs = Output(code, msg)
        if content_type is not None:
            outputs.add_property("content-type", content_type)

        if ts_out is None:
            outputs.message = "No content"
        else:
            val = ts_out[0]
            if isinstance(val, torch.Tensor):
                buff = io.BytesIO()
                torch.save(val, buff)
                buff.seek(0)
                val = buff.read()

            outputs.add(val)

        return outputs
