#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from collections import OrderedDict

from djl_python.rolling_batch.rolling_batch import RollingBatch
from djl_python.rolling_batch.vllm_inference_mixin import VllmInferenceMixin
from lmi_dist.api import Request
from lmi_dist.init_engine import engine_from_args
from lmi_dist.api import Request
from vllm import EngineArgs, SamplingParams

from djl_python.properties_manager.lmi_dist_v2_rb_properties import LmiDistRbProperties

DTYPE_MAPPER = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "auto": "auto"
}


class LmiDistRollingBatch(RollingBatch, VllmInferenceMixin):
    """
    LmiDistRollingBatch connects handler to LmiDist backend engine. It receives new
    requests from the handler and sends them to the backend when space is available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    def __init__(self, model_id_or_path: str, properties: dict, **kwargs):
        """
        Initializes the LmiDistRollingBatch.

        :param model_id_or_path (str): Currently unused since there is a copy inside properties
        :param properties (dict): other properties of the model, such as decoder strategy
        """

        # TODO: check if VllmRbProperties can be used here
        self.lmi_dist_configs = LmiDistRbProperties(**properties)

        super().__init__(
            waiting_steps=self.lmi_dist_configs.waiting_steps,
            output_formatter=self.lmi_dist_configs.output_formatter)

        args = EngineArgs(
            model=self.lmi_dist_configs.model_id_or_path,
            tensor_parallel_size=self.lmi_dist_configs.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[self.lmi_dist_configs.dtype],
            seed=0,
            max_model_len=self.lmi_dist_configs.max_model_len,
            enforce_eager=self.lmi_dist_configs.enforce_eager,
            gpu_memory_utilization=self.lmi_dist_configs.gpu_memory_utilization,
            max_num_batched_tokens=self.lmi_dist_configs.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.lmi_dist_configs.trust_remote_code,
            load_format=self.lmi_dist_configs.load_format,
            quantization=self.lmi_dist_configs.quantize,
            revision=self.lmi_dist_configs.revision)
        
        self.engine = engine_from_args(args)
        self.request_cache = OrderedDict()


    def _add_request_to_engine(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        lmi_dist_request = Request(id=request_id, prompt=prompt, sampling_params=sampling_params)
        self.engine.add_request(lmi_dist_request)

    def reset(self):
        """
        Aborts all requests
        """
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = OrderedDict()
        super().reset()

    def translate_vllm_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that VLLM recognizes.

        :param parameters (dict): Parameters pertaining to a specific request

        :return: The same parameters dict, but with VLLM style parameter names.
        """
        parameters.pop('seed', None)
        parameters.pop('do_sample', None)
        if "max_new_tokens" in parameters.keys():
            parameters["max_tokens"] = parameters.pop("max_new_tokens")
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos")
        return parameters


    def inference(self, input_data, parameters):
        """
        Adds new requests and gets output tokens from the backend.

        :param input_data (list[str]): List of input prompts.
        :param parameters (list[str]): List of settings pertaining to each request.

        :return results (list): List of dictionaries, one for each request, that contain output tokens and other data.
        """
        return self.inference_impl(input_data, parameters)

    def preprocess_requests(self, requests):
        """
        Currently not applicable for VLLM.
        """
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
    