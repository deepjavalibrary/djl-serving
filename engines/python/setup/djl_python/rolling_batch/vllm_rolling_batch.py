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
from collections import OrderedDict

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import random_uuid

from djl_python.rolling_batch.vllm_rolling_batch_base import VllmRollingBatchBase, DTYPE_MAPPER
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties


class VLLMRollingBatch(VllmRollingBatchBase):
    """
    VLLMRollingBatch connects the handler to the backend (VLLM inference). It receives new
    requests from the handler and sends them to the backend when there is space available in the batch.
    It also gets any new tokens from the backend and sends them back to the handler.
    """

    # TODO: Make properties is the only parameter, after refactoring all rolling batch handlers
    def __init__(self, model_id_or_path: str, properties: dict,
                 **kwargs) -> None:
        """
        Initializes the VLLMRollingBatch.

        :param model_id_or_path: Currently unused since there is a copy inside properties
        :param properties: other properties of the model, such as decoder strategy
        """
        engine_config = VllmRbProperties(**properties)
        super().__init__(engine_config)
        self.init_engine()

    def init_engine(self):
        """
        Initializes vllm engine
        """
        args = EngineArgs(
            model=self.engine_config.model_id_or_path,
            tensor_parallel_size=self.engine_config.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[self.engine_config.dtype],
            seed=0,
            max_model_len=self.engine_config.max_model_len,
            enforce_eager=self.engine_config.enforce_eager,
            gpu_memory_utilization=self.engine_config.gpu_memory_utilization,
            max_num_batched_tokens=self.engine_config.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.engine_config.trust_remote_code,
            load_format=self.engine_config.load_format,
            quantization=self.engine_config.quantize,
            draft_model=self.engine_config.speculative_draft_model,
            speculate_length=self.engine_config.speculative_length,
            draft_model_tp_size=self.engine_config.draft_model_tp_size,
            revision=self.engine_config.revision)
        self.engine = LLMEngine.from_engine_args(args)

    def reset(self) -> None:
        """
        Aborts all requests
        """
        for key in self.request_cache.keys():
            self.engine.abort_request(key)
        self.request_cache = OrderedDict()
        super().reset()

    def add_request(self, request_id: str, prompt: str,
                    sampling_params: SamplingParams):
        """
        Adds request to the engine
        """
        self.engine.add_request(request_id, prompt, sampling_params)

    def translate_to_engine_params(self, parameters: dict) -> dict:
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that VLLM recognizes.

        :param parameters: Parameters pertaining to a specific request

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

    def get_request_id(self, request):
        """
        Get request id that will be set to backend engine request
        """
        return random_uuid()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for VLLM.
        """
        raise NotImplementedError("Not implemented for vLLM rolling batcher")
