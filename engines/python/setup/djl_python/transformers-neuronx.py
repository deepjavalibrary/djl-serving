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
import tempfile
import os
import logging
import torch

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers_neuronx import dtypes
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.gptneox.model import GPTNeoXForSampling
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.opt.model import OPTForSampling
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.bloom.model import BloomForSampling
from transformers_neuronx.module import save_pretrained_split
from djl_python import Input, Output
from djl_python.stable_diffusion_inf2 import StableDiffusionService
from djl_python.streaming_utils import StreamingUtils

model = None

DTYPE_MAPPER = {"fp32": "f32", "fp16": "f16"}

SUPPORTED_MODEL_TYPES = {"opt", "gpt2", "gptj", "gpt_neox", "llama", "bloom"}

MODEL_TYPE_TO_MODEL= {
    "opt": OPTForSampling,
    "gpt2": GPT2ForSampling,
    "gptj": GPTJForSampling,
    "gpt_neox": GPTNeoXForSampling,
    "llama": LlamaForSampling,
    "bloom": BloomForSampling
}


class TransformersNeuronXService(object):

    def __init__(self) -> None:
        self.initialized = False
        self.batch_size = None
        self.model_id_or_path = None
        self.tensor_parallel_degree = None
        self.model = None
        self.tokenizer = None
        self.enable_streaming = None
        self.download_dir = None
        self.amp = None
        self.unroll = None
        self.n_positions = None
        self.model_type = None

    def convert_dtype(self, dtype, model_type):
        if model_type == "opt":
            for block in self.model.model.decoder.layers:
                block.self_attn.to(dtype)
                block.fc1.to(dtype)
                block.fc2.to(dtype)
            self.model.lm_head.to(dtype)
        elif model_type == "bloom":
            for block in self.model.transformer.h:
                block.self_attention.to(dtype)
                block.mlp.to(dtype)
            self.model.lm_head.to(dtype)
        elif model_type in ["gpt2", "gptj"]:
            for block in self.model.transformer.h:
                block.attn.to(dtype)
                block.mlp.to(dtype)
            self.model.lm_head.to(dtype)
        elif model_type == "gpt_neox":
            for block in self.model.gpt_neox.layers:
                block.attention.to(dtype)
                block.mlp.to(dtype)
            self.model.embed_out.to(dtype)
        elif model_type == "llama":
            for block in self.model.model.layers:
                block.self_attn.q_proj.to(dtype)
                block.self_attn.k_proj.to(dtype)
                block.self_attn.v_proj.to(dtype)
                block.self_attn.o_proj.to(dtype)
                block.mlp.gate_proj.to(dtype)
                block.mlp.down_proj.to(dtype)
                block.mlp.up_proj.to(dtype)
            self.model.lm_head.to(dtype)
        else:
            raise AttributeError(f"Model architecture format is not implemented")

    @staticmethod
    def is_inf2_model(model_path):
        return os.path.exists(os.path.join(model_path, "verify"))

    def model_amp_match(self, model_path):
        if not self.is_inf2_model(model_path):
            return False
        with open(os.path.join(model_path, "verify")) as f:
            return self.amp in f.read()

    def init_load_path(self, model_type):
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        folder = f"inf2_{model_type}_{self.amp}"
        if not path:
            path = tempfile.gettempdir()
        if not os.path.exists(os.path.join(path, folder)):
            os.mkdir(os.path.join(path, folder))
        self.download_dir = os.path.join(path, folder)
        return self.download_dir

    def convert_model(self, model_type, load_path):
        if self.is_inf2_model(load_path):
            return
        self.model = self.load_hf_model()
        logging.info("Start model conversion to INF2 format...")
        dtype = dtypes.to_torch_dtype(self.amp)
        self.convert_dtype(dtype, model_type)
        logging.info(f"Saving INF2 model to {load_path} ...")
        save_pretrained_split(self.model, load_path)
        with open(os.path.join(load_path, "verify"), "w") as f:
            f.writelines(f"{model_type}-converted-{self.amp}")

    def get_load_path(self, model_type):
        if self.is_inf2_model(self.model_id_or_path):
            load_path = self.model_id_or_path
        elif self.download_dir:
            load_path = self.download_dir
        else:
            load_path = self.init_load_path(model_type)
        return load_path

    def load_hf_model(self):
        logging.info(f"Start loading the model {self.model_id_or_path}...")
        return AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path, low_cpu_mem_usage=True)

    def load_inf2_model(self, model_type, load_path):
        return MODEL_TYPE_TO_MODEL[model_type].from_pretrained(
            load_path,
            batch_size=self.batch_size,
            amp=self.amp,
            tp_degree=self.tensor_parallel_degree,
            n_positions=self.n_positions,
            unroll=self.unroll)

    def load_model(self, model_type):
        load_path = self.get_load_path(model_type)
        self.convert_model(model_type, load_path)
        self.model = self.load_inf2_model(model_type, load_path)
        if model_type == "gpt2":
            self.model._load_compiled_artifacts(load_path)
            self.model.to_neuron()
            self.model._save_compiled_artifacts(load_path)
        else:
            self.model.to_neuron()

    def initialize(self, properties):
        self.batch_size = int(properties.get("batch_size", 1))
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.enable_streaming = properties.get("enable_streaming", None)
        if self.enable_streaming and self.enable_streaming.lower() == "false":
            self.enable_streaming = None
        dtype = properties.get("dtype", "fp32")
        self.n_positions = int(properties.get("n_positions", 128))
        self.unroll = properties.get("unroll", None)
        if dtype not in DTYPE_MAPPER:
            raise ValueError(f"{dtype} data type not supported!")
        self.amp = DTYPE_MAPPER[dtype]
        model_config = AutoConfig.from_pretrained(self.model_id_or_path)
        if model_config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"{model_config.model_type} type not supported for model {self.model_id_or_path}"
                f"Supported model arch: {SUPPORTED_MODEL_TYPES}")
        self.model_type = model_config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path, padding_side="left")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.load_model(model_config.model_type)

        # HuggingFace compatible generate model
        self.model = HuggingFaceGenerationModelAdapter(model_config,
                                                       self.model)
        self.initialized = True

    def infer(self, inputs):
        try:
            input_map = inputs.get_as_json()
            input_text = input_map.pop("inputs", input_map)
            parameters = input_map.pop("parameters", {})
            if isinstance(input_text, str):
                input_text = [input_text]
            if len(input_text) != self.batch_size:
                raise ValueError(
                    f"{self.batch_size} batch size not equal to {len(input_text)} prompt size"
                )
            outputs = Output()
            model_kwargs = {}

            if self.enable_streaming:
                outputs.add_property("content-type", "application/jsonlines")
                if self.enable_streaming == "huggingface":
                    outputs.add_stream_content(
                        StreamingUtils.use_hf_default_streamer(
                            self.model, self.tokenizer, input_text, None,
                            **model_kwargs))
                else:
                    stream_generator = StreamingUtils.get_stream_generator(
                        "transformers-neuronx")
                    model_kwargs["engine"] = "transformers-neuronx"
                    outputs.add_stream_content(
                        stream_generator(self.model, self.tokenizer,
                                         input_text, "cpu", **model_kwargs))
                return outputs

            encoded_inputs = self.tokenizer.batch_encode_plus(
                input_text, return_tensors="pt", padding=True)
            use_sample = parameters.pop("use_sample", None)
            if use_sample:
                # TODO: Watch transformer-neuronx release for fix on gpt-neox generate functionality
                output_tokens = self.model.sample(encoded_inputs.input_ids, sequence_length=self.n_positions,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  eos_token_id=self.tokenizer.eos_token_id,
                                                  **parameters)
            else:
                output_tokens = self.model.generate(
                    input_ids=encoded_inputs.input_ids,
                    attention_mask=encoded_inputs.attention_mask,
                    **parameters)
            generated_text = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True)

            return Output().add([{
                "generated_text": s
            } for s in generated_text])

        except Exception as e:
            logging.exception("TransformerNeuronX inference failed")
            outputs = Output().error((str(e)))
        return outputs


_service = TransformersNeuronXService()


def handle(inputs: Input):
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionService()
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warm up the model on startup
        return None

    return _service.infer(inputs)