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
import logging
from collections import OrderedDict

from lmi_dist.api import Request
from lmi_dist.init_engine import engine_from_args
from vllm import EngineArgs, SamplingParams

from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    get_speculative_decoding_metrics_record, update_request_cache_with_output,
    supports_speculative_decoding, DTYPE_MAPPER, FINISH_REASON_MAPPER)
from djl_python.properties_manager.lmi_dist_v2_rb_properties import LmiDistV2RbProperties


class LmiDistRollingBatch(RollingBatch):
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
        self.lmi_dist_config = LmiDistV2RbProperties(**properties)
        super().__init__(
            waiting_steps=self.lmi_dist_config.waiting_steps,
            output_formatter=self.lmi_dist_config.output_formatter)
        self.supports_speculative_decoding = supports_speculative_decoding()
        engine_kwargs = {}
        if self.supports_speculative_decoding:
            engine_kwargs[
                "draft_model"] = self.vllm_configs.speculative_draft_model
            engine_kwargs[
                "speculate_length"] = self.vllm_configs.speculative_length
            engine_kwargs[
                "draft_model_tp_size"] = self.vllm_configs.draft_model_tp_size
        args = EngineArgs(
            model=self.lmi_dist_config.model_id_or_path,
            tensor_parallel_size=self.lmi_dist_config.tensor_parallel_degree,
            dtype=DTYPE_MAPPER[self.lmi_dist_config.dtype],
            seed=0,
            max_model_len=self.lmi_dist_config.max_model_len,
            enforce_eager=self.lmi_dist_config.enforce_eager,
            gpu_memory_utilization=self.lmi_dist_config.gpu_memory_utilization,
            max_num_batched_tokens=self.lmi_dist_config.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.lmi_dist_config.trust_remote_code,
            load_format=self.lmi_dist_config.load_format,
            quantization=self.lmi_dist_config.quantize,
            revision=self.lmi_dist_config.revision,
            **engine_kwargs)
        self.engine = engine_from_args(args)
        self.request_cache = OrderedDict()
        self.model_type = getattr(kwargs.get("model_config", None),
                                  "model_type", None)

    def reset(self) -> None:
        """
        Aborts all requests
        """
        self.engine.reset(self.request_cache.keys())
        self.request_cache = OrderedDict()
        super().reset()

    def translate_lmi_dist_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that lmidist_v2 recognizes.

        :param parameters (dict): Parameters pertaining to a specific request

        :return: The same parameters dict, but with lmi-dist style parameter names.
        """
        do_sample = parameters.pop('do_sample', False)
        if do_sample and "temperature" not in parameters.keys():
            parameters["temperature"] = 1.0
        else:
            parameters["temperature"] = 0.0
        if "seed" in parameters.keys():
            parameters["seed"] = int(parameters["seed"])
        if "max_new_tokens" in parameters.keys():
            parameters["max_tokens"] = parameters.pop("max_new_tokens")
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos")
        return parameters

    @stop_on_any_exception
    def inference(self, input_data: list[str], parameters: list[dict]) -> list:
        """
        Adds new requests and gets output tokens from the backend.

        :param input_data: List of input prompts.
        :param parameters: List of settings pertaining to each request.

        :return results: List of dictionaries, one for each request, that contain output tokens and other data.
        """
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)
        # step 0: register new requests to engine
        for request in new_requests:
            request_id = str(request.id)
            params = self.translate_lmi_dist_params(request.parameters)
            sampling_params = SamplingParams(**params)
            lmi_dist_request = Request(id=request_id,
                                       prompt=request.input_text,
                                       sampling_params=sampling_params)
            self.engine.add_request(lmi_dist_request)
            self.request_cache[request_id] = {
                "curr_length": 0,
                "text": "",
                "cumulative_logprob": 0.0,
                "log_prob": 0.0,
                "finished": False,
                "finish_reason": None
            }
        request_outputs = self.engine.step()

        # step 1: put result to cache
        for request_output in request_outputs:
            self.request_cache = update_request_cache_with_output(
                self.request_cache, request_output)
            # Record SD metrics
            completion_output = request_output.outputs[0]
            if self.lmi_dist_config.record_acceptance_rate and request_output.finished:
                if self.supports_speculative_decoding and completion_output.acceptance_history:
                    record = get_speculative_decoding_metrics_record(
                        completion_output, request_output)
                    logging.info(f"Speculative Decoding {record}")
                else:
                    logging.warning(
                        f"Ignoring logging speculative decoding metrics")

        # step 2: send result back
        finished_id = []
        for (key, cache), request in zip(self.request_cache.items(),
                                         self.active_requests):
            finish_reason = None
            if cache["finished"]:
                finished_id.append(key)
                finish_reason = FINISH_REASON_MAPPER.get(
                    cache["finish_reason"], None)
            text = cache["text"][cache["curr_length"]:]
            if len(text) > 0:
                # token id is not determined since there could be multiple token comes at the same time
                # only return the last one
                token = Token(cache['id'], text, cache["log_prob"])
                request.set_next_token(token, self.output_formatter,
                                       cache["finished"], finish_reason)
            else:
                request.set_next_token("", self.output_formatter,
                                       cache["finished"], finish_reason)
            cache["curr_length"] = len(cache["text"])

        # step 3: clean finished requests
        for key in finished_id:
            self.request_cache.pop(key)

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for lmi-dist-v2.
        """
        raise NotImplementedError(
            "Not implemented for lmidist_v2 rolling batcher")
