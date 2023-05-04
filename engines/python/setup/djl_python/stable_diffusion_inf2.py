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
import os
import torch
import torch.nn as nn
import torch_neuronx
from djl_python.inputs import Input
from djl_python.outputs import Output
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.cross_attention import CrossAttention


class UNetWrap(nn.Module):

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                cross_attention_kwargs=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              return_dict=False)
        return out_tuple


class NeuronUNet(nn.Module):

    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0], )),
                               encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp32":
        return torch.float32
    elif dtype == "fp16":
        return torch.float16
    raise ValueError(
        f"Invalid data type: {dtype}. DeepSpeed currently only supports fp16 for stable diffusion"
    )


def get_attention_scores(self, query, key, attn_mask):
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if query.size() == key.size():
        attention_scores = cust_badbmm(key, query.transpose(-1, -2))

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores,
                                                      dim=1).permute(0, 2, 1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(query, key.transpose(-1, -2))

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)

    return attention_probs


def cust_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled


class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.model_id_or_path = None
        self.data_type = None
        self.tensor_parallel_degree = None

    def initialize(self, properties: dict):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise, we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 2))
        self.data_type = get_torch_dtype_from_str(
            properties.get("dtype", "fp32"))
        kwargs = {"torch_dtype": self.data_type}
        if "use_auth_token" in properties:
            kwargs["use_auth_token"] = properties["use_auth_token"]

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id_or_path, **kwargs)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config)

        # Replace original cross-attention module with custom cross-attention module for better performance
        CrossAttention.get_attention_scores = get_attention_scores

        if os.path.exists(os.path.join(self.model_id_or_path,
                                       "compiled_model")):
            logging.info("Loading pre-compiled model")
            self.load_compiled(
                os.path.join(self.model_id_or_path, "compiled_model"))
        else:
            self.runtime_compile()

        if "save_compiled_model" in properties:
            self.save_compiled(
                os.path.join(properties.get("save_compiled_model"),
                             "compiled_model"))

        device_ids = [idx for idx in range(self.tensor_parallel_degree)]
        self.pipeline.unet.unetwrap = torch_neuronx.DataParallel(
            self.pipeline.unet.unetwrap,
            device_ids,
            set_dynamic_batching=False)

        self.initialized = True

    def runtime_compile(self):
        logging.warning(
            "Runtime compilation is not recommended, please precompile the model"
        )
        logging.info("Model compilation started...")
        COMPILER_WORKDIR_ROOT = "/tmp/neuron_compiler"
        self.pipeline.unet = NeuronUNet(UNetWrap(self.pipeline.unet))

        sample_1b = torch.randn([1, 4, 64, 64])
        timestep_1b = torch.tensor(999).float().expand((1, ))
        encoder_hidden_states_1b = torch.randn([1, 77, 1024])
        example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

        logging.info("Compiling UNET...")
        self.pipeline.unet.unetwrap = torch_neuronx.trace(
            self.pipeline.unet.unetwrap,
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
            compiler_args=[
                "--internal-hlo2penguin-options=--expand-batch-norm-training",
                "--policy=3"
            ])

        logging.info("Compiling post_quant_conv_in...")
        # Compile vae post_quant_conv
        post_quant_conv_in = torch.randn([1, 4, 64, 64])
        self.pipeline.vae.post_quant_conv = torch_neuronx.trace(
            self.pipeline.vae.post_quant_conv,
            post_quant_conv_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                          'vae_post_quant_conv'))

        logging.info("Compiling VAE Decoder...")
        # Compile vae decoder
        decoder_in = torch.randn([1, 4, 64, 64])
        self.pipeline.vae.decoder = torch_neuronx.trace(
            self.pipeline.vae.decoder,
            decoder_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                          'vae_decoder'),
            compiler_args=[
                "--tensorizer-options=--max-dma-access-free-depth=3",
                "--policy=3"
            ])

    def save_compiled(self, saved_dir):
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        # save compiled unet
        unet_filename = os.path.join(saved_dir, 'unet.pt')
        torch.jit.save(self.pipeline.unet.unetwrap, unet_filename)
        # Save the compiled vae post_quant_conv
        post_quant_conv_filename = os.path.join(saved_dir,
                                                'vae_post_quant_conv.pt')
        torch.jit.save(self.pipeline.vae.post_quant_conv,
                       post_quant_conv_filename)
        # Save the compiled vae decoder
        decoder_filename = os.path.join(saved_dir, 'vae_decoder.pt')
        torch.jit.save(self.pipeline.vae.decoder, decoder_filename)

    def load_compiled(self, saved_dir):
        post_quant_conv_filename = os.path.join(saved_dir,
                                                'vae_post_quant_conv.pt')
        self.pipeline.vae.post_quant_conv = torch.jit.load(
            post_quant_conv_filename)
        decoder_filename = os.path.join(saved_dir, 'vae_decoder.pt')
        self.pipeline.vae.decoder = torch.jit.load(decoder_filename)
        self.pipeline.unet = NeuronUNet(UNetWrap(self.pipeline.unet))
        unet_filename = os.path.join(saved_dir, 'unet.pt')
        self.pipeline.unet.unetwrap = torch.jit.load(unet_filename)

    def infer(self, inputs: Input):
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
