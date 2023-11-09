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

from seq_scheduler.lm_block import HuggingfaceBlock, BloomBlock, FalconBlock
from seq_scheduler.search_config import SearchConfig
from seq_scheduler.seq_batch_scheduler import SeqBatchScheduler
from collections import namedtuple, defaultdict
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import torch

MODEL_TYPE_2_BLOCK = {'bloom': BloomBlock, 'falcon': FalconBlock}
DEFAULT_SEARCH_ALGORITHM = 'greedy'
# https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#efficient-inference-on-a-single-gpu
FLASH_2_SUPPORTED_MODELS = {
    "LlamaForCausalLM", "RWForCausalLM", "FalconForCausalLM"
}


def enable_flash():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
    return False


class SchedulerRollingBatch(RollingBatch):

    def __init__(self, model_id_or_path, device, properties, **kwargs):
        """
        Initializes the rolling batch scheduler.

        :param model_id_or_path: model id or path
        :param device: model loaded device
        :param properties: other properties of the model, such as decoder strategy
        :param kwargs passed while loading the model
        """

        super().__init__(device, **kwargs)
        self._init_model_and_tokenizer(model_id_or_path,
                                       device=device,
                                       properties=properties,
                                       multi_gpu=properties.get(
                                           'multi_gpu', None),
                                       **kwargs)
        self._init_scheduler(properties)

    @stop_on_any_exception
    def inference(self, input_text, parameters):
        """
        Performs prefill and decode operations for the batch.

        :param input_text: List of input texts for each request in a batch
        :param parameters: List of kwargs for each request in a batch
        :return: generated batch decoded tokens
        """
        batch_size = len(input_text)
        new_requests = self.get_new_requests(input_text, parameters,
                                             batch_size)

        preprocessed_new_requests = self.preprocess_requests(new_requests)
        self._prefill_and_decode(preprocessed_new_requests)
        return self.postprocess_results()

    def preprocess_requests(self, requests):
        Requests = namedtuple(
            'Requests',
            ['input_texts', 'search_configs', 'request_ids', 'prompts'])
        new_requests = Requests(defaultdict(list), defaultdict(list),
                                defaultdict(list), defaultdict(dict))

        for request in requests:
            parameters = request.parameters
            search_algorithm = parameters.get('decoding_strategy',
                                              self.search_algorithm)

            # TODO: This is not needed when search algorithm automatically chosen for the user.
            if str(parameters.get(
                    "do_sample",
                    self.search_config.sampling)).lower() == "true":
                search_algorithm = "sample"

            new_requests.input_texts[search_algorithm].append(
                request.input_text)

            if "cached_prompt" in parameters:
                new_requests.prompts[search_algorithm][
                    request.id] = parameters.pop("cached_prompt")

            search_config = self._construct_search_config(parameters)
            new_requests.search_configs[search_algorithm].append(search_config)
            new_requests.request_ids[search_algorithm].append(request.id)

        return new_requests

    def _init_model_and_tokenizer(self,
                                  model_id_or_path,
                                  device=None,
                                  multi_gpu=None,
                                  properties=None,
                                  **kwargs):
        if "waiting_steps" in kwargs:
            kwargs.pop("waiting_steps")
        if "output_formatter" in kwargs:
            kwargs.pop("output_formatter")
        self.config = AutoConfig.from_pretrained(model_id_or_path, **kwargs)
        architectures = self.config.architectures
        if architectures and architectures[0].endswith(
                "ForConditionalGeneration"):
            raise ValueError('Seq2Seq model is not supported by scheduler')
        else:
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            if 'device_map' in kwargs:
                device_map = kwargs.pop('device_map')
            elif device:
                if isinstance(device, str) or isinstance(device, int):
                    device_map = device
                elif isinstance(device, torch.device):
                    device_map = 'auto' if device.type == 'cuda' else 'cpu'

            if architectures and architectures[
                    0] in FLASH_2_SUPPORTED_MODELS and enable_flash():
                if properties.get("disable_flash_attn",
                                  "true").lower() != 'true':
                    kwargs['use_flash_attention_2'] = True

            if "lmi_dist_sharding" == multi_gpu:
                if 'neox' in model_id_or_path:
                    try:
                        from lmi_dist.models.gpt_neox import GPTNeoxSharded
                        from lmi_dist.utils import download_and_convert_weights

                        download_and_convert_weights(model_id_or_path)
                        self.model = GPTNeoxSharded(model_id_or_path)
                    except ImportError:
                        print(
                            f"Running {model_id_or_path} requires package lmi_dist."
                        )
                else:
                    raise Exception(
                        f"{model_id_or_path} with lmi_dist_sharding is currently unsupported."
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path, device_map=device_map, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_scheduler(self, properties):
        lm_block_cls = MODEL_TYPE_2_BLOCK.get(self.config.model_type,
                                              HuggingfaceBlock)
        self.lm_block = lm_block_cls(self.model)
        self.search_config = SearchConfig(
            eos_token_id=self.tokenizer.eos_token,
            pad_token_id=self.tokenizer.pad_token)
        self.search_algorithm = properties.get('decoding_strategy',
                                               DEFAULT_SEARCH_ALGORITHM)
        self.scheduler = SeqBatchScheduler(
            self.lm_block,
            self.search_algorithm,
            self.search_config,
            max_sparsity=float(properties.get(
                'max_sparsity',
                0.33)),  # a threshold to limit the max padding sparsity
            max_splits=int(properties.get(
                'max_splits',
                3)))  # a threshold to limit the max number of batch splits

    def _prefill_and_decode(self, new_requests):

        for search_algorithm in new_requests.request_ids.keys():
            request_ids = new_requests.request_ids[search_algorithm]
            if request_ids:
                input_texts = new_requests.input_texts[search_algorithm]
                search_configs = new_requests.search_configs[search_algorithm]
                prompt_ids = self._get_prompt_ids(
                    new_requests.prompts[search_algorithm])
                prompt_ids = prompt_ids if prompt_ids else None
                # Prefills search states for each request and merges to the existing batch
                self.scheduler.add_request(
                    input_ids=self._get_input_ids(input_texts=input_texts),
                    request_uids=_get_request_ids_tensor(
                        request_ids=request_ids),
                    search_algorithm=search_algorithm,
                    search_configs=search_configs,
                    kv_cache_prompt_ids=prompt_ids)

        # Decoding step. Generates a token for all the requests in a batch.
        generated_token_ids, request_ids, exit_req_ids = self.scheduler.inference_call(
        )

        if self.scheduler and self.scheduler.seq_batchers and self.scheduler.seq_batchers[
                0]:
            seq_len = self.scheduler.seq_batchers[
                0].seq_len - self.scheduler.seq_batchers[0].offsets[0]

        # TODO: Deleting the finished results here temporarily
        for request_id in exit_req_ids:
            if request_id in self.scheduler.results:
                del self.scheduler.results[request_id]

        generated_tokens = self.tokenizer.batch_decode(generated_token_ids)

        for request_id, generated_token, request in zip(
                request_ids, generated_tokens, self.active_requests):
            is_last_token = (request_id in exit_req_ids)
            request.set_next_token(f" {generated_token}",
                                   self.output_formatter,
                                   last_token=is_last_token)

    def _get_input_ids(self, input_texts):
        input_ids = self.tokenizer(input_texts,
                                   return_tensors="pt",
                                   padding=True).input_ids
        input_ids = input_ids.to(self.model.device)
        return input_ids

    def _get_prompt_ids(self, prompts):
        prompt_ids = {}
        for req_id, prompt in prompts.items():
            prompt_id = self.tokenizer(prompt,
                                       return_tensors="pt",
                                       padding=True).input_ids
            prompt_id = prompt_id.view(1, -1)
            prompt_ids[req_id] = prompt_id
        return prompt_ids

    def _construct_search_config(self, parameters):
        use_lru_kv_cache = str(
            parameters.get(
                'use_lru_kv_cache',
                self.search_config.use_lru_kv_cache)).lower() == "true"
        do_sample = str(
            parameters.get('do_sample',
                           self.search_config.sampling)).lower() == "true"
        return SearchConfig(
            max_new_tokens=parameters.get('max_new_tokens',
                                          self.search_config.max_new_seqlen),
            top_k=parameters.get('top_k', self.search_config.topk),
            penalty_alpha=parameters.get('penalty_alpha',
                                         self.search_config.alpha),
            num_beams=parameters.get('num_beams', self.search_config.beam),
            do_sample=do_sample,
            top_p=parameters.get('top_p', self.search_config.topk),
            temperature=parameters.get('temperature',
                                       self.search_config.temperature),
            use_lru_kv_cache=use_lru_kv_cache,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id)


def _get_request_ids_tensor(request_ids):
    request_ids_tensor = torch.tensor(request_ids)
    request_ids_tensor = torch.reshape(request_ids_tensor,
                                       (len(request_ids), 1))
    return request_ids_tensor


def _calculate_req_id_counter(scheduler):
    if scheduler:
        request_ids = scheduler.get_request_ids()
        if request_ids:
            return request_ids[-1] + 1
    return 0
