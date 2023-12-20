#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import logging
from djl_python.inputs import Input
from djl_python.outputs import Output
from io import BytesIO
from PIL import Image
from djl_python.neuron_utils.model_loader import OptimumStableDiffusionLoader
from djl_python.properties_manager.sd_inf2_properties import StableDiffusionNeuronXProperties


class StableDiffusionNeuronXService(object):

    def __init__(self):
        self.config = None
        self.pipeline = None
        self.pipeline_loader = None
        self.initialized = False

    def get_pipeline_kwargs(self):
        pipeline_kwargs = {"torch_dtype": self.config.dtype}
        if self.config.use_auth_token is not None:
            pipeline_kwargs["use_auth_token"] = self.config.use_auth_token
        self.pipeline_loader = OptimumStableDiffusionLoader(config=self.config,
                                                            compile_model=True)
        return pipeline_kwargs

    def initialize(self, properties: dict):
        self.config = StableDiffusionNeuronXProperties(**properties)
        pipeline_kwargs = self.get_pipeline_kwargs()
        self.pipeline = self.pipeline_loader.load_pipeline(**pipeline_kwargs)
        self.initialized = True

    def partition(self, properties: dict):
        self.config = StableDiffusionNeuronXProperties(**properties)
        pipeline_kwargs = self.get_pipeline_kwargs()
        self.pipeline = self.pipeline_loader.partition(
            self.config.save_mp_checkpoint_path, **pipeline_kwargs)
        self.initialized = True

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                result = self.pipeline(prompt, **params)
            elif content_type and content_type.startswith("text/"):
                prompt = inputs.get_as_string()
                result = self.pipeline(prompt)
            else:
                init_image = Image.open(BytesIO(
                    inputs.get_as_bytes())).convert("RGB")
                request = inputs.get_as_json("json")
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                result = self.pipeline(prompt, image=init_image, **params)

            img = result.images[0]
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_img = buf.getvalue()
            outputs = Output().add(byte_img).add_property(
                "content-type", "image/png")

        except Exception as e:
            logging.exception("Neuron inference failed")
            outputs = Output().error(str(e))
        return outputs
