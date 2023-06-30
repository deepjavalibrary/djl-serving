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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers_neuronx import dtypes
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.opt.model import OPTForSampling
from djl_python import Input, Output
from djl_python.stable_diffusion_inf2 import StableDiffusionService
from djl_python.streaming_utils import StreamingUtils

model = None

DTYPE_MAPPER = {"fp32": "f32", "fp16": "f16"}

SUPPORTED_MODEL_TYPES = {"opt", "gpt2", "gptj"}


class TransformersNeuronXService(object):

    def __init__(self) -> None:
        self.initialized = False
        self.batch_size = None
        self.model_id_or_path = None
        self.tensor_parallel_degree = None
        self.model = None
        self.tokenizer = None
        self.enable_streaming = None

    def convert_opt(self, amp):
        logging.warning(
            "Model conversion is a slow process to do in runtime, please consider convert it"
            " Ahead-of-Time next time")
        logging.info(f"Start loading the model {self.model_id_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path, low_cpu_mem_usage=True)
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        if not path:
            path = tempfile.gettempdir()

        load_path = tempfile.mkdtemp(dir=path, prefix="inf2_")
        logging.info("Start model conversion to INF2 format...")
        dtype = dtypes.to_torch_dtype(amp)
        for block in self.model.model.decoder.layers:
            block.self_attn.to(dtype)
            block.fc1.to(dtype)
            block.fc2.to(dtype)
        self.model.lm_head.to(dtype)
        logging.info(f"Saving INF2 model to {load_path} ...")
        save_pretrained_split(self.model, load_path)
        with open(os.path.join(load_path, "verify"), "w") as f:
            f.writelines("opt-converted")
        return load_path

    def convert_gpt(self, amp, gpt_type="gpt2"):
        logging.warning(
            "Model conversion is a slow process to do in runtime, please consider convert it"
            " Ahead-of-Time next time")
        logging.info(f"Start loading the model {self.model_id_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path, low_cpu_mem_usage=True)
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        if not path:
            path = tempfile.gettempdir()

        load_path = tempfile.mkdtemp(dir=path, prefix="inf2_")
        logging.info("Start model conversion to INF2 format...")
        dtype = dtypes.to_torch_dtype(amp)
        for block in self.model.transformer.h:
            block.attn.to(dtype)
            block.mlp.to(dtype)
        self.model.lm_head.to(dtype)
        logging.info(f"Saving to INF2 model to {load_path} ...")
        self.model.save_pretrained(load_path, max_shard_size="100GB")
        with open(os.path.join(load_path, "verify"), "w") as f:
            f.writelines(f"{gpt_type}-converted")
        return load_path

    def load_opt(self, amp, unroll, n_positions):
        load_path = self.model_id_or_path
        if not os.path.exists(os.path.join(load_path, "verify")):
            load_path = self.convert_opt(amp)
        self.model = OPTForSampling.from_pretrained(
            load_path,
            batch_size=self.batch_size,
            amp=amp,
            tp_degree=self.tensor_parallel_degree,
            n_positions=n_positions,
            unroll=unroll)
        self.model.to_neuron()

    def load_gpt2(self, amp, unroll, n_positions):
        load_path = self.model_id_or_path
        if not os.path.exists(os.path.join(load_path, "verify")):
            load_path = self.convert_gpt(amp, gpt_type="gpt2")
        self.model = GPT2ForSampling.from_pretrained(
            load_path,
            batch_size=self.batch_size,
            amp=amp,
            tp_degree=self.tensor_parallel_degree,
            n_positions=n_positions,
            unroll=unroll)
        # load GPT2 compiled artifacts if exist
        self.model._load_compiled_artifacts(load_path)
        self.model.to_neuron()
        # save GPT2 compiled artifacts
        self.model._save_compiled_artifacts(load_path)

    def load_gptj(self, amp, unroll, n_positions):
        load_path = self.model_id_or_path
        if not os.path.exists(os.path.join(load_path, "verify")):
            load_path = self.convert_gpt(amp, gpt_type="gptj")
        self.model = GPTJForSampling.from_pretrained(
            load_path,
            batch_size=self.batch_size,
            amp=amp,
            tp_degree=self.tensor_parallel_degree,
            n_positions=n_positions,
            unroll=unroll)
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
        n_positions = int(properties.get("n_positions", 128))
        unroll = properties.get("unroll", None)
        if dtype not in DTYPE_MAPPER:
            raise ValueError(f"{dtype} data type not supported!")
        amp = DTYPE_MAPPER[dtype]
        model_config = AutoConfig.from_pretrained(self.model_id_or_path)
        if model_config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"{model_config.model_type} type not supported for model {self.model_id_or_path}"
                f"Supported model arch: {SUPPORTED_MODEL_TYPES}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
        self.tokenizer.padding_side = 'left'
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "opt" == model_config.model_type:
            self.load_opt(amp, unroll, n_positions)
        elif "gpt2" == model_config.model_type:
            self.load_gpt2(amp, unroll, n_positions)
        elif "gptj" == model_config.model_type:
            self.load_gptj(amp, unroll, n_positions)
        # HuggingFace compatible generate model
        self.model = HuggingFaceGenerationModelAdapter(model_config,
                                                       self.model)
        self.initialized = True

    def infer(self, inputs):
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
        output_tokens = self.model.generate(
            input_ids=encoded_inputs.input_ids,
            attention_mask=encoded_inputs.attention_mask,
            **parameters)
        generated_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True)

        return Output().add([{
            "generated_text": s
        } for s in generated_text])


_service = TransformersNeuronXService()


def handle(inputs: Input):
    global _service
    if not _service.initialized:
        if "use_stable_diffusion" in inputs.get_properties():
            _service = StableDiffusionService()
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return _service.infer(inputs)
