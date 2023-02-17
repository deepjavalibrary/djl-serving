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
from PIL import Image


def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Invalid data type: {dtype}. DeepSpeed currently only supports fp16 for stable diffusion")


class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.tensor_parallel_degree = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        # If option.s3url is used, the directory is stored in model_id
        # If option.s3url is not used but model_id is present, we download from hub
        # Otherwise we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get("model_dir")
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        enable_cuda_graph = False
        if properties.get("enable_cuda_graph", "false").lower() == "true":
            if self.tensor_parallel_degree > 1:
                raise ValueError("enable_cuda_graph optimization can only be used with tensor_parallel_degree=1")
            enable_cuda_graph = True
        self.ds_config = self._get_ds_config_for_dtype(self.data_type, enable_cuda_graph)

        if os.path.exists(self.model_id_or_path):
            config_file = os.path.join(self.model_id_or_path, "model_index.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"{self.model_id_or_path} does not contain a model_index.json."
                    f"This is required for loading stable diffusion models from local storage"
                )

        # DS 0.8.0 only supports fp16 and by this point we have validated dtype
        kwargs = {"torch_dtype": torch.float16, "revision": "fp16"}

        pipeline = DiffusionPipeline.from_pretrained(self.model_id_or_path, **kwargs)
        pipeline.to(f"cuda:{self.device}")
        deepspeed.init_distributed()
        engine = deepspeed.init_inference(getattr(pipeline, "model", pipeline),
                                          **self.ds_config)

        if hasattr(pipeline, "model"):
            pipeline.model = engine

        self.pipeline = pipeline
        self.initialized = True

    def _get_ds_config_for_dtype(self, dtype, cuda_graph):
        # This is a workaround due to 2 issues with DeepSpeed 0.7.5
        # 1. No kernel injection is available for stable diffusion using fp32 (kernels only written for fp16)
        # 2. Changes in our bf16 fork raise an error, but the original deepspeed codebase defaults to fp16
        #    when dtype is not set explicitly. We need to be explicit here with this config
        ds_config = {
            "enable_cuda_graph": cuda_graph,
            "dtype": dtype,
            "tensor_parallel": {"tp_size": self.tensor_parallel_degree},
            "replace_method": "auto",
            "replace_with_kernel_inject": True,
        }
        return ds_config

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
                init_image = Image.open(BytesIO(inputs.get_as_bytes())).convert("RGB")
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
