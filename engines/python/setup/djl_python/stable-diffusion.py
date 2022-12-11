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
import logging
import os
import torch
from diffusers import DiffusionPipeline
import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional
from io import BytesIO


def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    elif dtype is None:
        return None
    else:
        raise ValueError(f"Invalid data type: {dtype}")


class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.logger = logging.getLogger()
        self.model_dir = None
        self.model_id = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.tensor_parallel_degree = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        self.model_dir = properties.get("model_dir")
        self.model_id = properties.get("model_id")
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        self.ds_config = {
            "replace_with_kernel_inject": True,
            # TODO: Figure out why cuda graph doesn't work for stable diffusion via DS
            "enable_cuda_graph": False,
            "replace_method": "auto",
            "dtype": self.data_type,
            "mp_size": self.tensor_parallel_degree
        }

        if not self.model_id:
            config_file = os.path.join(self.model_dir, "model_index.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"model_dir: {self.model_dir} does not contain a model_index.json."
                    f"This is required for loading stable diffusion models from local storage"
                )
            self.model_id = self.model_dir

        kwargs = {}
        if self.data_type == torch.float16:
            kwargs["torch_dtype"] = torch.float16
            kwargs["revision"] = "fp16"

        pipeline = DiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        pipeline.to(f"cuda:{self.device}")
        deepspeed.init_distributed()
        engine = deepspeed.init_inference(getattr(pipeline, "model", pipeline),
                                          **self.ds_config)

        if hasattr(pipeline, "model"):
            pipeline.model = engine

        self.pipeline = pipeline
        self.initialized = True

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters")
                result = self.pipeline(prompt, **params)
            else:
                prompt = inputs.get_as_string()
                result = self.pipeline(prompt)

            img = result.images[0]
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_img = buf.getvalue()
            outputs = Output().add(byte_img).add_property(
                "content-type", "image/png")

        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            outputs = Output().error(str(e))
        return outputs


_service = StableDiffusionService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)
