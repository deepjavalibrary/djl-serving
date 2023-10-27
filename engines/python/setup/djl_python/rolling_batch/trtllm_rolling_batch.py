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
import os
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception

class TRTLLMRollingBatch(RollingBatch):
	def __init__(self, model_id_or_path, device, properties, **kwargs):
		"""
		Initializes the TRTLLMRollingBatch.
		:param model_id_or_path: model id or path
		:param properties: other properties of the model, such as decoder strategy
		"""
		super().__init__(-1, **kwargs)
		self.fix_config(os.path.join(model_id_or_path, "tensorrt_llm/config.pbtxt"))
		self.core = tritontoolkit.init_triton(model_id_or_path)
		self.model = self.core.load_model("tensorrt_llm")
		self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left", revision=kwargs.get('revision', None))
		self.request_cache = OrderedDict()

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
		"""
		Stops all current requests and resets state of rolling batch portion of handler
		"""
		for key in self.request_cache.keys():
			continue
			# todo stop the asynchronous inference
		self.request_cache = OrderedDict()
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
		payload = {
			"input_ids": input_ids_data,
			"input_lengths": input_lengths_data,
			"request_output_len": request_output_len_data,
			"streaming": streaming_data,
		}
		return payload

	def inference(self, input_data, parameters):
		batch_size = len(input_data)
		# add pending requests to active requests list
		new_requests = self.get_new_requests(input_data, parameters, batch_size)
		# do we take care of prefill/decode in here or does trtllm take care of that?
		# register new active requests
		for request in new_requests:
			result = self.model.inference_async(self.get_sample_payload(request.input_text, request.parameters))
			self.request_cache[request_id] = {"response_obj":result, "curr_length":0, "text":"", "finished":False}

		# obtain new tokens in all active requests
		finished_ids = set()
		for request in self.active_requests:
			cached_request = request_cache[request.id]
			data, complete = cached_request["response_obj"].get_result()
			output_id = data["output_ids"].squeeze().tolist()
			output_text = " " + self.tokenizer.decode(output_id)
			cached_request["curr_length"] += 1
			cached_request["text"] += token
			if(complete): # placeholder: "finished"
				finished_ids.add(request_id)
				cached_request["finished"] = True

		# remove finished requests
		for finished_id in finished_ids:
			del self.request_cache[finished_id]

		return self.postprocess_results()