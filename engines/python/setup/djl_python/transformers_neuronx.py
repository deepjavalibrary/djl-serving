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
import json
import shutil
import tempfile
import os
import logging
import time

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.gptneox.model import GPTNeoXForSampling
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.opt.model import OPTForSampling
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.bloom.model import BloomForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.config import NeuronConfig, QuantizationConfig
from djl_python import Input, Output
from djl_python.encode_decode import decode, encode
from djl_python.rolling_batch.neuron_rolling_batch import NeuronRollingBatch
from djl_python.stable_diffusion_inf2 import StableDiffusionService
from djl_python.streaming_utils import StreamingUtils

model = None

DTYPE_MAPPER = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}

SUPPORTED_MODEL_TYPES = {"opt", "gpt2", "gptj", "gpt_neox", "llama", "bloom"}

MODEL_TYPE_TO_MODEL = {
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
        self.trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE",
                                                "FALSE").lower() == 'true'
        self.revision = None
        self.rolling_batch = None
        self.load_in_8bit = False

    def init_load_path(self, model_type):
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        folder = f"inf2_{model_type}_{self.amp}"
        if not path:
            path = tempfile.gettempdir()
        if os.path.exists(os.path.join(path, folder)):
            shutil.rmtree(os.path.join(path, folder))
        os.mkdir(os.path.join(path, folder))
        self.download_dir = os.path.join(path, folder)
        return self.download_dir

    def convert_model(self, load_path):
        self.model = self.load_hf_model()
        logging.info(f"Saving INF2 model to {load_path} ...")
        save_pretrained_split(self.model, load_path)

    def get_load_path(self, model_type):
        if self.download_dir:
            load_path = self.download_dir
        else:
            load_path = self.init_load_path(model_type)
        return load_path

    def load_hf_model(self):
        logging.info(f"Start loading the model {self.model_id_or_path}...")
        return AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            low_cpu_mem_usage=True)

    def load_inf2_model(self, model_type, load_path):
        if self.load_in_8bit:
            neuron_config = NeuronConfig()
            neuron_config.quant = QuantizationConfig(quant_dtype='s8',
                                                     dequant_dtype=self.amp)
            return MODEL_TYPE_TO_MODEL[model_type].from_pretrained(
                load_path,
                batch_size=self.batch_size,
                amp=self.amp,
                tp_degree=self.tensor_parallel_degree,
                n_positions=self.n_positions,
                neuron_config=neuron_config,
                unroll=self.unroll)
        return MODEL_TYPE_TO_MODEL[model_type].from_pretrained(
            load_path,
            batch_size=self.batch_size,
            amp=self.amp,
            tp_degree=self.tensor_parallel_degree,
            n_positions=self.n_positions,
            unroll=self.unroll)

    def load_model(self, model_type):
        load_path = self.get_load_path(model_type)
        self.convert_model(load_path)
        self.model = self.load_inf2_model(model_type, load_path)
        logging.info(f"LLM sharding and compiling Started ...")
        start = time.time()
        # TODO: workaround on Neuron Compiler bug for SM
        path = os.getcwd()
        os.chdir("/tmp")
        if model_type == "gpt2":
            self.model._load_compiled_artifacts(load_path)
            self.model.to_neuron()
            self.model._save_compiled_artifacts(load_path)
        else:
            self.model.to_neuron()
        os.chdir(path)
        elapsed = time.time() - start
        logging.info(f"LLM sharding and compiling completed with {elapsed}s")

    def initialize(self, properties):
        # Neuron recommendation for transformersneuronx speedup
        os.environ["NEURON_CC_FLAGS"] = os.environ[
            "NEURON_CC_FLAGS"] + " --model-type=transformer"
        if "neuron_optimize_level" in properties:
            level = properties.get("neuron_optimize_level")
            os.environ["NEURON_CC_FLAGS"] = os.environ[
                "NEURON_CC_FLAGS"] + f" -O{level}"
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
        if "trust_remote_code" in properties:
            self.trust_remote_code = properties.get(
                "trust_remote_code").lower() == "true"
        if "revision" in properties:
            self.revision = properties.get("revision")
        if "load_in_8bit" in properties:
            self.load_in_8bit = properties.get(
                "load_in_8bit").lower() == 'true'
        model_config = AutoConfig.from_pretrained(self.model_id_or_path,
                                                  revision=self.revision)
        if model_config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"{model_config.model_type} type not supported for model {self.model_id_or_path}"
                f"Supported model arch: {SUPPORTED_MODEL_TYPES}")
        self.model_type = model_config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            padding_side="left")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.load_model(model_config.model_type)

        # HuggingFace compatible generate model
        self.model = HuggingFaceGenerationModelAdapter(model_config,
                                                       self.model)
        self.initialized = True
        if "rolling_batch" in properties:
            self.rolling_batch = NeuronRollingBatch(self.model, self.tokenizer,
                                                    self.batch_size,
                                                    self.n_positions)

    def parse_input(self, inputs):
        input_data = []
        input_size = []
        parameters = []
        errors = {}
        batch = inputs.get_batches()
        first = True
        for i, item in enumerate(batch):
            try:
                content_type = item.get_property("Content-Type")
                input_map = decode(item, content_type)
                _inputs = input_map.pop("inputs", input_map)
                if first or self.rolling_batch:
                    parameters.append(input_map.pop("parameters", {}))
                    first = False
                else:
                    param = input_map.pop("parameters", {})
                    if parameters[0] != param:
                        logging.warning(
                            f"expected param: {parameters}, actual: {param}")
                        raise ValueError(
                            "In order to enable dynamic batching, all input batches must have the same parameters"
                        )
                if isinstance(_inputs, list):
                    input_data.extend(_inputs)
                    input_size.append(len(_inputs))
                else:
                    input_data.append(_inputs)
                    input_size.append(1)
            except Exception as e:  # pylint: disable=broad-except
                logging.exception(f"Parse input failed: {i}")
                errors[i] = str(e)

        return input_data, input_size, parameters, errors, batch

    def inference(self, inputs):
        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs)
        outputs = Output()

        if self.rolling_batch:
            if inputs.get_property("reset_rollingbatch"):
                self.rolling_batch.reset()
            result = self.rolling_batch.inference(input_data, parameters)
            idx = 0
            for i in range(len(batch)):
                err = errors.get(i)
                if err:
                    err = json.dumps({"code": 424, "error": err})
                    err = json.dumps({"data": err, "last": True})
                    outputs.add(err, key="data", batch_index=i)
                else:
                    outputs.add(result[idx], key="data", batch_index=i)
                    idx += 1

            content_type = self.rolling_batch.get_content_type()
            if content_type:
                outputs.add_property("content-type", content_type)
            return outputs

        parameters = parameters[0]
        model_kwargs = {}

        prompt_size = len(input_data)
        if prompt_size > self.batch_size:
            raise ValueError(
                f"Batch size {prompt_size} beyond the max_batch size the model can support {self.batch_size}"
            )

        for i in range(prompt_size, self.batch_size):
            input_data.append(self.tokenizer.eos_token)

        # clean KV cache
        self.model.reset_generation()
        if self.enable_streaming:
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.enable_streaming == "huggingface":
                outputs.add_stream_content(
                    StreamingUtils.use_hf_default_streamer(
                        self.model, self.tokenizer, input_data, None,
                        **model_kwargs))
            else:
                stream_generator = StreamingUtils.get_stream_generator(
                    "transformers-neuronx")
                model_kwargs["engine"] = "transformers-neuronx"
                outputs.add_stream_content(
                    stream_generator(self.model, self.tokenizer, input_data,
                                     "cpu", **model_kwargs))
            return outputs

        encoded_inputs = self.tokenizer.batch_encode_plus(input_data,
                                                          return_tensors="pt",
                                                          padding=True)
        use_sample = parameters.pop("use_sample", None)
        if use_sample:
            # TODO: Watch transformer-neuronx release for fix on gpt-neox generate functionality
            output_tokens = self.model.sample(
                encoded_inputs.input_ids,
                sequence_length=self.n_positions,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **parameters)
        else:
            output_tokens = self.model.generate(
                input_ids=encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask,
                **parameters)
        prediction = self.tokenizer.batch_decode(output_tokens,
                                                 skip_special_tokens=True)

        # trim the input based on the actual size
        prediction = prediction[:prompt_size]
        prediction = [{"generated_text": s} for s in prediction]

        offset = 0
        for i, item in enumerate(batch):
            content_type = item.get_property("Content-Type")
            accept = item.get_property("Accept")
            if not accept:
                content_type = content_type if content_type else "application/json"
                accept = content_type if content_type.startswith(
                    "tensor/") else "application/json"
            elif "*/*" in accept:
                accept = "application/json"

            err = errors.get(i)
            if err:
                encode(outputs,
                       err,
                       accept,
                       key=inputs.get_content().key_at(i))
            else:
                encode(outputs,
                       prediction[offset:offset + input_size[i]],
                       accept,
                       key=inputs.get_content().key_at(i))
                offset += input_size[i]

        outputs.add_property("content-type", "application/json")

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

    return _service.inference(inputs)
