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

import fastertransformer as ft
from transformers import AutoConfig
from djl_python import Input, Output
import logging
from typing import Optional


class FasterTransformerService(object):

    t5_default_args = dict(inputs_embeds=None,
                           beam_width=1,
                           max_seq_len=200,
                           top_k=1,
                           top_p=0.0,
                           beam_search_diversity_rate=0.0,
                           temperature=1.0,
                           len_penalty=0.0,
                           repetition_penalty=1.0,
                           presence_penalty=None,
                           min_length=0,
                           random_seed=0,
                           is_return_output_log_probs=False,
                           is_return_cum_log_probs=False,
                           is_return_cross_attentions=False,
                           bad_words_list=None,
                           stop_words_list=None)

    gpt_default_args = dict(beam_width=1,
                            top_k=1,
                            top_p=0.0,
                            temperature=1,
                            repetition_penalty=1.0,
                            random_seed=0,
                            len_penalty=0,
                            min_length=0,
                            return_output_length=0,
                            return_cum_log_probs=0)

    def __init__(self) -> None:
        self.initialized = False
        self.tensor_parallel_degree = -1
        self.pipeline_parallel_degree = -1
        self.dtype = None
        self.model_id_or_path = None
        self.model = None
        self.is_t5 = False
        self.use_triton = None
        self.enable_streaming = None

    def initialize_properties(self, properties):
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        self.pipeline_parallel_degree = int(
            properties.get("pipeline_parallel_degree", 1))
        self.dtype = properties.get("data_type", self.dtype)
        self.dtype = properties.get("dtype", "fp32")
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.use_triton = properties.get("engine", "Python") == "Python"
        self.enable_streaming = properties.get("enable_streaming", None)
        if self.enable_streaming and self.enable_streaming.lower() != "false":
            if properties.get("engine", "Python") != "Python":
                raise ValueError(
                    "Please use Python engine for streaming use case")
            self.use_triton = True

    def initialize(self, properties):
        self.initialize_properties(properties)
        self.model = self.load_model()
        model_config = AutoConfig.from_pretrained(self.model_id_or_path)
        self.is_t5 = model_config.model_type == "t5"
        self.initialized = True

    def load_model(self):
        logging.info(f"Loading model: {self.model_id_or_path}")
        return ft.init_inference(self.model_id_or_path,
                                 self.tensor_parallel_degree,
                                 self.pipeline_parallel_degree,
                                 self.dtype,
                                 use_triton=self.use_triton,
                                 do_streaming=self.enable_streaming)

    @staticmethod
    def param_mapper(parameters: dict):
        if "max_new_tokens" in parameters:
            logging.warning(
                "max_new_tokens not supported for FasterTransformer,"
                " will replace to max_length instead.")
            parameters["max_seq_len"] = parameters.pop("max_new_tokens")
        if "max_length" in parameters:
            parameters["max_seq_len"] = parameters.pop("max_length")
        if "num_beams" in parameters:
            parameters["beam_width"] = parameters.pop("num_beams")
        if "length_penalty" in parameters:
            parameters["len_penalty"] = parameters.pop("length_penalty")
        return parameters

    def inference(self, inputs: Input):
        # TODO: Add support for more content types
        input_data = []
        input_size = []
        parameters = {}
        batches = inputs.get_batches()
        first = True
        for item in batches:
            input_map = item.get_as_json()
            input_text = input_map.pop("inputs", input_map)
            if isinstance(input_text, str):
                input_text = [input_text]
            input_size.append(len(input_text))
            input_data.extend(input_text)
            if first:
                parameters = input_map.pop("parameters", {})
                first = False
            else:
                if parameters != input_map.pop("parameters", {}):
                    return Output().error(
                        "In order to enable dynamic batching, all input batches must have the same parameters"
                    )

        parameters = self.param_mapper(parameters)
        max_length = parameters.pop("max_length", 50)
        output_len = parameters.pop("max_seq_len", max_length)
        if self.use_triton:
            output_length = [output_len] * len(input_data)
            if self.enable_streaming:
                outputs = Output()
                outputs.add_property("content-type", "application/jsonlines")
                outputs.add_stream_content(
                    self.model.stream_generate(input_data, output_length,
                                               **parameters))
                return outputs
            result = self.model.pipeline_generate(input_data, output_length,
                                                  **parameters)
        else:
            if self.is_t5:
                result = self.model.pipeline_generate(input_data, **parameters)
            else:
                beam_width = parameters.pop("beam_width", 1)
                # TODO: remove after fixes in FT python package
                result = self.model.pipeline_generate(
                    input_data,
                    batch_size=len(input_data),
                    output_len=output_len,
                    beam_width=beam_width,
                    **parameters)

        offset = 0
        outputs = Output()
        outputs.add_property("content-type", "application/json")
        for i in range(inputs.get_batch_size()):
            generated_text = [{
                "generated_text": s
            } for s in result[offset:offset + input_size[i]]]
            outputs.add(generated_text, key=inputs.get_content().key_at(i))

        return outputs


_service = FasterTransformerService()


def partition(inputs: Input):
    properties = inputs.get_properties()
    _service.initialize_properties(properties)
    ft.save_checkpoint(_service.model_id_or_path,
                       _service.tensor_parallel_degree,
                       _service.pipeline_parallel_degree,
                       properties.get("save_mp_checkpoint_path"),
                       _service.dtype)


def handle(inputs: Input) -> Optional[Output]:

    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return _service.inference(inputs)
