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
from djl_python.properties_manager.tnx_properties import TransformerNeuronXProperties

from djl_python.properties_manager.properties import StreamingEnum

model = None

SUPPORTED_MODEL_TYPES = {"opt", "gpt2", "gptj", "gpt_neox", "llama", "bloom"}

MODEL_TYPE_TO_MODEL = {
    "opt": OPTForSampling,
    "gpt2": GPT2ForSampling,
    "gptj": GPTJForSampling,
    "gpt_neox": GPTNeoXForSampling,
    "llama": LlamaForSampling,
    "bloom": BloomForSampling
}


class NeuronXSampleAdapter(HuggingFaceGenerationModelAdapter):

    def __init__(self, _config, _model):
        super().__init__(_config, _model)
        self.model_type = _config.model_type
        self.sample_options = ["start_ids", "top_k"]
        if self.model_type == "llama":
            self.sample_options = self.sample_options + [
                "top_p", "eos_token_override", "temperature", "streamer"
            ]

    def neuron_sample(self, *args, **kwargs):
        sample_kwargs = self.simple_sample_parser(**kwargs)
        return self.model.sample(*args, **sample_kwargs)

    def simple_sample_parser(self, **kwargs):
        parsed_kwargs = dict()
        for key in self.sample_options:
            if key in kwargs:
                parsed_kwargs[key] = kwargs[key]
        return parsed_kwargs


class TransformersNeuronXService(object):

    def __init__(self) -> None:
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.download_dir = None
        self.model_type = None
        self.rolling_batch = None
        self.tnx_configs = None

    def init_load_path(self, model_type):
        path = os.environ.get("SERVING_DOWNLOAD_DIR")
        folder = f"inf2_{model_type}_{self.tnx_configs.amp}"
        if not path:
            path = tempfile.gettempdir()
        if os.path.exists(os.path.join(path, folder)):
            shutil.rmtree(os.path.join(path, folder))
        os.mkdir(os.path.join(path, folder))
        self.download_dir = os.path.join(path, folder)
        return self.download_dir

    def get_load_path(self, model_type):
        if self.download_dir:
            load_path = self.download_dir
        else:
            load_path = self.init_load_path(model_type)
        return load_path

    def load_hf_model(self):
        logging.info(
            f"Start loading the model {self.tnx_configs.model_id_or_path}...")
        return AutoModelForCausalLM.from_pretrained(
            self.tnx_configs.model_id_or_path,
            trust_remote_code=self.tnx_configs.trust_remote_code,
            revision=self.tnx_configs.revision,
            low_cpu_mem_usage=True)

    def get_model_specific_kwargs(self, model_type):
        model_kwargs = {
            "batch_size": self.tnx_configs.batch_size,
            "amp": self.tnx_configs.amp,
            'tp_degree': self.tnx_configs.tensor_parallel_degree,
            "n_positions": self.tnx_configs.n_positions,
            "unroll": self.tnx_configs.unroll
        }
        if model_type == "llama":
            model_kwargs[
                'context_length_estimate'] = self.tnx_configs.context_length_estimate
        return model_kwargs

    def load_inf2_model_from_disk(self, model_type, load_path):
        if not self.tnx_configs.load_split_model:
            logging.info(f"Saving INF2 model to {load_path} ...")
            save_pretrained_split(self.model, load_path)
        model_kwargs = self.get_model_specific_kwargs(model_type)
        if self.tnx_configs.load_in_8bit:
            neuron_config = NeuronConfig()
            neuron_config.quant = QuantizationConfig(
                quant_dtype='s8', dequant_dtype=self.tnx_configs.amp)
            return MODEL_TYPE_TO_MODEL[model_type].from_pretrained(
                load_path, neuron_config=neuron_config, **model_kwargs)
        return MODEL_TYPE_TO_MODEL[model_type].from_pretrained(
            load_path, **model_kwargs)

    def load_inf2_model_from_memory(self, model_type):
        model_kwargs = self.get_model_specific_kwargs(model_type)
        if self.tnx_configs.load_in_8bit:
            neuron_config = NeuronConfig()
            neuron_config.quant = QuantizationConfig(
                quant_dtype='s8', dequant_dtype=self.tnx_configs.amp)
            model = MODEL_TYPE_TO_MODEL[model_type](
                self.model.config, neuron_config=neuron_config, **model_kwargs)
        else:
            model = MODEL_TYPE_TO_MODEL[model_type](self.model.config,
                                                    **model_kwargs)
        model.load_state_dict_low_memory(self.model.state_dict())
        return model

    def load_model(self, model_type):
        if not self.tnx_configs.load_split_model:
            self.model = self.load_hf_model()
            load_path = self.get_load_path(model_type)
        else:
            load_path = self.tnx_configs.model_id_or_path
        if self.tnx_configs.low_cpu_mem_usage or self.tnx_configs.load_split_model:
            logging.info("Transferring weights from HF to INF2 through disk")
            self.model = self.load_inf2_model_from_disk(model_type, load_path)
        else:
            logging.info("Transferring weights from HF to INF2 in-memory")
            self.model = self.load_inf2_model_from_memory(model_type)
        logging.info(f"LLM sharding and compiling Started ...")
        start = time.time()
        # TODO: workaround on Neuron Compiler bug for SM
        path = os.getcwd()
        os.chdir("/tmp")
        self.model.to_neuron()
        os.chdir(path)
        elapsed = time.time() - start
        logging.info(
            f"SysHealth: LLM sharding and compilation latency: {elapsed} secs")

    def initialize(self, properties):
        # Neuron recommendation for transformersneuronx speedup
        os.environ["NEURON_CC_FLAGS"] = os.environ[
            "NEURON_CC_FLAGS"] + " --model-type=transformer"
        self.tnx_configs = TransformerNeuronXProperties(**properties)
        model_config = AutoConfig.from_pretrained(
            self.tnx_configs.model_id_or_path,
            revision=self.tnx_configs.revision)
        if model_config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"{model_config.model_type} type not supported for model {self.tnx_configs.model_id_or_path}"
                f"Supported model arch: {SUPPORTED_MODEL_TYPES}")
        self.model_type = model_config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tnx_configs.model_id_or_path,
            trust_remote_code=self.tnx_configs.trust_remote_code,
            revision=self.tnx_configs.revision,
            padding_side="left")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.load_model(model_config.model_type)

        # HuggingFace compatible generate model and Neuron custom sample method
        self.model = NeuronXSampleAdapter(model_config, self.model)
        self.initialized = True
        if self.tnx_configs.rolling_batch != "disable":
            self.rolling_batch = NeuronRollingBatch(
                self.model, self.tokenizer,
                self.tnx_configs.max_rolling_batch_size,
                self.tnx_configs.n_positions)

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
        if prompt_size > self.tnx_configs.batch_size:
            raise ValueError(
                f"Batch size {prompt_size} beyond the max_batch size the model can support {self.tnx_configs.batch_size}"
            )

        for i in range(prompt_size, self.tnx_configs.batch_size):
            input_data.append(self.tokenizer.eos_token)

        # clean KV cache
        self.model.reset_generation()
        if self.tnx_configs.enable_streaming != StreamingEnum.false:
            if len(batch) > 1:
                raise NotImplementedError(
                    "Dynamic batch not supported for generic streaming")
            outputs.add_property("content-type", "application/jsonlines")
            if self.tnx_configs.enable_streaming == StreamingEnum.huggingface:
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
        use_sample = parameters.pop("use_sample", True)
        if use_sample:
            sample_length = parameters.pop("max_new_tokens",
                                           self.tnx_configs.n_positions)
            output_tokens = self.model.neuron_sample(encoded_inputs.input_ids,
                                                     sample_length,
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
