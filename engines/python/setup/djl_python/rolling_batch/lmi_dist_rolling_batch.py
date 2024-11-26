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
import os
from typing import List, Optional
from collections import OrderedDict, defaultdict

from lmi_dist.api import Request, RequestParams
from lmi_dist.arg_utils import VllmEngineArgs
from lmi_dist.init_engine import engine_from_args
from lmi_dist.seq2seq_engine import Seq2SeqPreprocessor
from vllm import SamplingParams
from vllm.utils import AtomicCounter

from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, filter_unused_generation_params
from djl_python.rolling_batch.rolling_batch_vllm_utils import (
    get_speculative_decoding_metrics_record, update_request_cache_with_output,
    supports_speculative_decoding, create_lora_request, get_lora_request,
    DTYPE_MAPPER, get_prompt_inputs)
from djl_python.telemetry import telemetry_manager
from djl_python.properties_manager.lmi_dist_rb_properties import LmiDistRbProperties

LMI_DIST_GENERATION_PARAMS = set(RequestParams().__struct_fields__)


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
        self.lmi_dist_config = LmiDistRbProperties(**properties)
        super().__init__(self.lmi_dist_config)
        self.supports_speculative_decoding = supports_speculative_decoding()
        engine_kwargs = {}
        if self.supports_speculative_decoding:
            engine_kwargs[
                "draft_model"] = self.lmi_dist_config.speculative_draft_model
            engine_kwargs[
                "speculate_length"] = self.lmi_dist_config.speculative_length
            engine_kwargs[
                "draft_model_tp_size"] = self.lmi_dist_config.draft_model_tp_size
        engine_args = VllmEngineArgs(
            model=self.lmi_dist_config.model_id_or_path,
            tensor_parallel_size=self.lmi_dist_config.tensor_parallel_degree,
            pipeline_parallel_size=self.lmi_dist_config.
            pipeline_parallel_degree,
            dtype=DTYPE_MAPPER[self.lmi_dist_config.dtype],
            seed=0,
            max_model_len=self.lmi_dist_config.max_model_len,
            max_num_seqs=self.lmi_dist_config.max_rolling_batch_size,
            enforce_eager=self.lmi_dist_config.enforce_eager,
            gpu_memory_utilization=self.lmi_dist_config.gpu_memory_utilization,
            max_num_batched_tokens=self.lmi_dist_config.
            max_rolling_batch_prefill_tokens,
            trust_remote_code=self.lmi_dist_config.trust_remote_code,
            load_format=self.lmi_dist_config.load_format,
            quantization=self.lmi_dist_config.quantize,
            enable_lora=self.lmi_dist_config.enable_lora,
            max_loras=self.lmi_dist_config.max_loras,
            max_lora_rank=self.lmi_dist_config.max_lora_rank,
            fully_sharded_loras=self.lmi_dist_config.fully_sharded_loras,
            lora_extra_vocab_size=self.lmi_dist_config.lora_extra_vocab_size,
            long_lora_scaling_factors=self.lmi_dist_config.
            long_lora_scaling_factors,
            lora_dtype=self.lmi_dist_config.lora_dtype,
            max_cpu_loras=self.lmi_dist_config.max_cpu_loras,
            revision=self.lmi_dist_config.revision,
            enable_chunked_prefill=self.lmi_dist_config.enable_chunked_prefill,
            cpu_offload_gb=self.lmi_dist_config.cpu_offload_gb_per_gpu,
            enable_prefix_caching=self.lmi_dist_config.enable_prefix_caching,
            disable_sliding_window=self.lmi_dist_config.disable_sliding_window,
            limit_mm_per_prompt=self.lmi_dist_config.limit_mm_per_prompt,
            use_passive_workers=self.lmi_dist_config.use_passive_workers,
            tokenizer_mode=self.lmi_dist_config.tokenizer_mode,
            **engine_kwargs)

        kwargs = {}
        logging.info(f"engine_args: {engine_args}, kwargs: {kwargs}")

        if self.lmi_dist_config.max_rolling_batch_prefill_tokens is None:
            logging.warning(
                "djl-serving/lmi has changed the default behavior for max_rolling_batch_prefill_tokens in 0.30.0 (lmi v12). "
                "Previously, when max_rolling_batch_prefill_tokens was unset, djl-serving would use a warmup prefill limit of 4096 tokens. "
                "This behavior differs from vLLM's default behavior, which (essentially) defaults to max_model_len. As a result of this change, "
                "model deployments that worked previously may fail due to higher memory requirements at model loading time for the warmup phase. "
                "For more information on this change, and guidance on what configurations to set, please see "
                "https://github.com/deepjavalibrary/djl-serving/tree/master/serving/docs/lmi/announcements/breaking_changes.md"
            )
        self.engine = engine_from_args(engine_args, **kwargs)
        self.request_cache = OrderedDict()
        self.lora_id_counter = AtomicCounter(0)
        self.lora_requests = {}
        self.is_mistral_tokenizer = self.lmi_dist_config.tokenizer_mode == 'mistral'
        self.is_t5_model = isinstance(self.engine.preprocessor,
                                      Seq2SeqPreprocessor)

    def reset(self) -> None:
        """
        Aborts all requests
        """
        self.engine.reset(self.request_cache.keys())
        self.request_cache = OrderedDict()
        super().reset()

    def get_tokenizer(self):
        if self.is_t5_model:
            return self.engine.preprocessor.tokenizer
        return self.engine.preprocessor.tokenizer.tokenizer

    def get_huggingface_model_config(self):
        # TODO: this is a hack right now to get the model config from the engine. We should expose this as
        # an interface method and retrieve it from there after v12
        return self.engine.preprocessor.model_config.hf_config if not self.is_t5_model else None

    def translate_lmi_dist_params(self, parameters: dict):
        """
        Helper function to convert DJL Serving parameter names to parameter names
        that lmi-dist recognizes.

        :param parameters (dict): Parameters pertaining to a specific request

        :return: The same parameters dict, but with lmi-dist style parameter names.
        """
        parameters["max_tokens"] = parameters.pop("max_new_tokens", 30)
        # If `do_sample` is not provided, force temperature=0.0, i.e. greedy
        # else set to user-provided value or default to 1.0
        if not parameters.pop('do_sample', False):
            parameters['temperature'] = 0.0
        else:
            parameters['temperature'] = parameters.get('temperature', 1.0)
        if "seed" in parameters.keys():
            parameters["seed"] = int(parameters["seed"])
        if "stop_sequences" in parameters.keys():
            parameters["stop"] = parameters.pop("stop_sequences")
        if "ignore_eos_token" in parameters.keys():
            parameters["ignore_eos"] = parameters.pop("ignore_eos_token")
        if "num_beams" in parameters.keys():
            parameters["best_of"] = parameters.pop("num_beams")
            parameters["use_beam_search"] = True
        if parameters.pop("decoder_input_details", False):
            parameters["prompt_logprobs"] = 1
        if "best_of" in parameters.keys():
            # if n is not explicitly set, we return `best_of` values sequences.
            if "n" not in "best_of":
                parameters["n"] = parameters["best_of"]
        if "top_n_tokens" in parameters.keys():
            parameters["logprobs"] = parameters.pop("top_n_tokens")
        else:
            parameters["logprobs"] = parameters.get("logprobs", 1)
        parameters = filter_unused_generation_params(
            parameters,
            LMI_DIST_GENERATION_PARAMS,
            "lmi-dist",
            remove_unused_params=True)
        return parameters

    @stop_on_any_exception
    def inference(self, new_requests: List[Request]) -> List:
        """
        Adds new requests and gets output tokens from the backend.

        :param new_requests: List of requests

        :return results: List of dictionaries, one for each request, that contain output tokens and other data.
        """
        self.add_new_requests(new_requests)
        # step 0: register new requests to engine
        new_lmi_dist_requests = []
        for request in new_requests:
            request_id = str(request.id)
            prompt_inputs = get_prompt_inputs(request)
            params = self.translate_lmi_dist_params(request.parameters)
            request_params = RequestParams(**params)
            lora_request_params = dict()
            if request.adapter is not None:
                adapter_name = request.adapter.get_property("name")
                lora_request_params["lora_request"] = get_lora_request(
                    adapter_name, self.lora_requests)
            # Constructing Request in lmi-dist library
            lmi_dist_request = Request(id=request_id,
                                       prompt=prompt_inputs,
                                       params=request_params,
                                       **lora_request_params)
            new_lmi_dist_requests.append(lmi_dist_request)
            self.request_cache[request_id] = {
                "request_output": request.request_output
            }
        if new_lmi_dist_requests:
            self.engine.add_requests(new_lmi_dist_requests)

        request_outputs = self.engine.step()

        # step 1: put result to cache
        for request_output in request_outputs:
            self.request_cache = update_request_cache_with_output(
                self.request_cache, request_output, self.get_tokenizer())
            # Record SD metrics
            completion_output = request_output.outputs[0]
            if (
                    self.lmi_dist_config.record_acceptance_rate
                    or self.lmi_dist_config.speculative_telemetry
            ) and self.lmi_dist_config.speculative_draft_model and request_output.finished:
                try:
                    if self.supports_speculative_decoding and hasattr(
                            completion_output, 'acceptance_history'):
                        record = get_speculative_decoding_metrics_record(
                            completion_output, request_output)
                        if self.lmi_dist_config.record_acceptance_rate:
                            logging.info(f"Speculative Decoding {record}")
                        if self.lmi_dist_config.speculative_telemetry and os.environ.get(
                                "SAGEMAKER_SECURE_MODE") == "true":
                            telemetry_manager.record_speculative(record)
                except:
                    logging.debug("SD telemetry collection failed, ignore")

        for request in self.active_requests:
            request_output = request.request_output
            if request_output.finished:
                request.last_token = True

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        """
        Currently not applicable for lmi-dist.
        """
        raise NotImplementedError(
            "Not implemented for lmidist rolling batcher")

    def add_lora(self,
                 lora_name: str,
                 lora_path: str,
                 long_lora_max_len: Optional[int] = None):
        """
        Add LoRA adapter.
        """
        lora_id = self.lora_id_counter.inc(1)
        lora_request = create_lora_request(lora_name,
                                           lora_id,
                                           lora_path,
                                           long_lora_max_len=long_lora_max_len)
        self.lora_requests[lora_request.lora_name] = lora_request
        return self.engine.add_lora(lora_request)

    def remove_lora(self, lora_name):
        """
        Remove LoRA adapter.
        """
        lora_request = get_lora_request(lora_name, self.lora_requests)
        return self.engine.remove_lora(lora_request.lora_int_id)

    def pin_lora(self, lora_name):
        """
        Pin LoRA adapter.
        """
        lora_request = get_lora_request(lora_name, self.lora_requests)

        # To pin an adapter, adapter has to be registered already (by calling add_lora()).
        # If trying to pin an adapter that is not registered, we will get "LoRA is not registered" error.
        # However, registered adapters are maintained by LRUCache
        # and may be evicted if the number of adapters exceed capacity (max_cpu_loras).
        # So there will be two scenarios:
        # 1) An adapter is evicted, call add_lora() is necessary to avoid error.
        # 2) An adapter is not evicted, call add_lora() is not necessary.
        # But since whether an adapter is evicted is not exposed outside of engine,
        # and add_lora() in this case will take negligible time, we will still call add_lora().
        loaded = self.engine.add_lora(lora_request)
        return loaded and self.engine.pin_lora(lora_request.lora_int_id)
