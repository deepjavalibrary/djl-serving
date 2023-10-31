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
import numpy as np
import tritontoolkit
from tritontoolkit import triton_pybind
from transformers import AutoTokenizer
import os, time
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception


class DetokenizedTritonResponse:
    def __init__(self, response, tokenizer):
        self.triton_response = response
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0
        self.all_input_ids = []

    def decode_token(self) -> str:
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:self.read_offset], skip_special_tokens=False
        )
        new_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:], skip_special_tokens=False
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.all_input_ids)
            return new_text
        else:
            return ""

    def fetch(self):
        result, complete = self.triton_response.get_result()
        self.all_input_ids.append(result["output_ids"].squeeze().tolist())
        return self.decode_token(), complete

class TRTLLMRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the VLLMRollingBatch.
        :param model_id_or_path: model id or path
        :param properties: other properties of the model, such as decoder strategy
        """
        super().__init__(-1, **kwargs)
        rank = int(os.environ.get("Rank", 0))
        if rank == 0:
            self.fix_config(os.path.join(model_id_or_path, "tensorrt_llm/config.pbtxt"))
        else:
            time.sleep(3 + 1 * rank)
        self.core = tritontoolkit.init_triton(model_id_or_path)
        self.model = self.core.load_model("tensorrt_llm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left",
                                                       revision=kwargs.get('revision', None))
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.request_cache = {}


    def fix_config(self, config_path):
        with open(config_path, "r") as f:
            configs = f.readlines()
            for idx, line in enumerate(configs):
                if "decoupled" in line:
                    configs[idx] = "  decoupled: True\n"
                if "gpt_model_path" in line:
                    base_name = os.path.abspath(os.path.dirname(config_path))
                    file_loc = os.path.join(base_name, "1")
                    configs[idx + 2] = f'    string_value: "{file_loc}"\n'
        with open(config_path, "w") as f:
            f.writelines(configs)

    def reset(self):
        for key, value in self.request_cache.items():
            triton_pybind.delete_request(value)
        self.request_cache.clear()
        super().reset()

    def get_sample_payload(self, input_text, parameters):
        input_tokens = self.tokenizer([input_text])
        input_ids = input_tokens["input_ids"]
        input_ids_data = np.array(input_ids, dtype=np.int32)
        input_lengths_data = np.array([[len(input_ids)]], dtype=np.int32)
        max_new_tokens = 64
        if "max_new_tokens" in parameters.keys():
            max_new_tokens = parameters.get("max_new_tokens")
        request_output_len_data = np.array([[max_new_tokens]], dtype=np.uint32)
        streaming_data = np.array([[True]], dtype=bool)
        pad_id_data = np.array([[self.tokenizer.pad_token_id]], dtype=np.uint32)
        end_id_data = np.array([[self.tokenizer.eos_token_id]], dtype=np.uint32)
        payload = {
            "input_ids": input_ids_data,
            "input_lengths": input_lengths_data,
            "request_output_len": request_output_len_data,
            "streaming": streaming_data,
            "pad_id": pad_id_data,
            "end_id": end_id_data,
        }
        if "temperature" in parameters.keys():
            temperature = parameters.get("temperature")
            payload["temperature"] = np.array([[temperature]], dtype=np.float32)
        if "min_length" in parameters.keys():
            min_length = parameters.get("min_length")
            payload["min_length"] = np.array([[min_length]], dtype=np.uint32)
        if "top_k" in parameters.keys():
            topk = parameters.get("top_k")
            payload["runtime_top_k"] = np.array([[topk]], dtype=np.uint32)
        if "top_p" in parameters.keys():
            topp = parameters.get("top_p")
            payload["runtime_top_p"] = np.array([[topp]], dtype=np.float32)
        if "repetition_penalty" in parameters.keys():
            repetition_penalty = parameters.get("repetition_penalty")
            payload["repetition_penalty"] = np.array([[repetition_penalty]], dtype=np.float32)
        if "seed" in parameters.keys():
            seed = int(parameters.get("seed"))
            payload["random_seed"] = np.array([[seed]], dtype=np.uint64)

        return payload

    @stop_on_any_exception
    def inference(self, input_data, parameters):
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new requests to engine
        for request in new_requests:
            result = self.model.inference_async(self.get_sample_payload(request.input_text, request.parameters))
            self.request_cache[request.id] = DetokenizedTritonResponse(result, self.tokenizer)
        # step 1: loop the active requests to send result
        for request in self.active_requests:
            trt_req = self.request_cache[request.id]
            output_text, complete = trt_req.fetch()
            request.set_next_token(output_text, self.output_formatter, complete)
            if complete:
                self.request_cache.pop(request.id)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
