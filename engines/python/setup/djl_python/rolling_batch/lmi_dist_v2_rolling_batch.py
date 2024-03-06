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

from lmi_dist.api import Request
from lmi_dist.init_engine import engine_from_args
from vllm import EngineArgs, SamplingParams

from djl_python.rolling_batch.vllm_rolling_batch_base import VllmRollingBatchBase, DTYPE_MAPPER
from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties


class LmiDistRollingBatch(VllmRollingBatchBase):
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

        engine_config = VllmRbProperties(**properties)
        super().__init__(engine_config)
        self.request_cache = OrderedDict()
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
            revision=self.engine_config.revision)
        self.engine = engine_from_args(args)

    def add_request(self, request_id: str, prompt: str,
                    sampling_params: SamplingParams):
        """
        Adds request to the engine
        """
        lmi_dist_request = Request(id=request_id,
                                   prompt=prompt,
                                   sampling_params=sampling_params)
        self.engine.add_request(lmi_dist_request)

    def translate_to_engine_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that lmidist_v2 recognizes.

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

    def get_request_id(self, request):
        """
        Get request id that will be set to backend engine request
        """
        return request.id

    def preprocess_requests(self, requests):
        """
        Currently not applicable for VLLM.
        """
        raise NotImplementedError(
            "Not implemented for lmidist_v2 rolling batcher")
