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

from collections import defaultdict
from typing import List, Dict
from dataclasses import dataclass
from djl_python import test_model, Input
from djl_python.request import Request
from djl_python.input_parser import format_input


@dataclass
class SimulationSchedule:
    prompts: List[str]
    params: List[Dict]
    reqs_to_prefill: List[int]
    wait_steps: List[int]


class NeuronRollingBatchGenerator:

    def __init__(self):
        self.rolling_batch = None
        self._req_id = 0
        # Store the results
        self.output_all = defaultdict(list)
        self.input_all = {}
        self.data_collector = []
        self.responses = []

        # Status variables
        self.input_str = []
        self.params = []
        self.req_ids = []

        # Spec_dec
        self.token_numbers = defaultdict(list)

    def init_neuron_service(self, properties: dict):
        from djl_python.transformers_neuronx import TransformersNeuronXService
        _service = TransformersNeuronXService()
        _service.initialize(properties)
        self.rolling_batch = _service.rolling_batch

    def get_req_id(self):
        req_id = self._req_id
        self._req_id = self._req_id + 1
        return req_id

    def collect_data(self, result):
        done_requests_indices = []
        for idx, item in enumerate(result):
            if len(self.data_collector) <= idx:
                self.data_collector.append(item["data"])
            else:
                self.data_collector[idx] += item["data"]
            if item['last']:
                done_requests_indices.append(idx)
        for idx in sorted(done_requests_indices, reverse=True):
            value = self.data_collector.pop(idx)
            self.responses.append(value)
            print(f"\nFinished request: {value}\n")
        return done_requests_indices

    def build_request(self, raw_input):
        inputs = test_model.create_json_request(raw_input)
        parsed_inputs = format_input(inputs)
        request = Request(parsed_inputs)
        request.id = self.get_req_id()
        return request

    def simulator(self, schedule: SimulationSchedule):
        assert len(schedule.prompts) == len(schedule.params)
        assert len(schedule.reqs_to_prefill) == len(schedule.wait_steps)
        zipped_requests = zip(schedule.prompts, schedule.params)
        all_requests = [{
            "inputs": prompt,
            "parameters": params
        } for prompt, params in zipped_requests]
        current_requests = []
        new_requests = []
        for batch_size, step in zip(schedule.reqs_to_prefill,
                                    schedule.wait_steps):
            for _ in range(batch_size):
                request = self.build_request(all_requests.pop(0))
                new_requests = [request] + new_requests
                current_requests.append(request)

            for i in range(step):
                if len(current_requests) == 0:
                    break
                generated_tokens = self.rolling_batch.inference(new_requests)
                new_requests.clear()
                finished_indices = self.collect_data(generated_tokens)
                for idx in sorted(finished_indices, reverse=True):
                    current_requests.pop(idx)
        while len(current_requests) > 0:
            generated_tokens = self.rolling_batch.inference(new_requests)
            finished_indices = self.collect_data(generated_tokens)
            for idx in sorted(finished_indices, reverse=True):
                current_requests.pop(idx)

    def step(self, step=20, input_str_delta=None, params_delta=None):
        if input_str_delta:
            begin_id = max(self.input_all.keys(), default=0) + 1
            req_ids_delta = list(
                range(begin_id, begin_id + len(input_str_delta)))

            self.input_str += input_str_delta
            self.params += params_delta
            self.req_ids += req_ids_delta
            for req_id, input_s, param in zip(req_ids_delta, input_str_delta,
                                              params_delta):
                self.input_all[req_id] = (input_s, param)

        iterator = range(step)
        for i in iterator:
            result = self.rolling_batch.inference(self.input_str, self.params)
            for res, req_id in zip(result, self.req_ids):
                self.output_all[req_id].append(res['data'])
                self.token_numbers[req_id].append(res.get('step_token_num', 1))
            self.req_ids = [
                req_id for req_id, res in zip(self.req_ids, result)
                if not res['last']
            ]
            self.input_str = [
                s for s, res in zip(self.input_str, result) if not res['last']
            ]
            self.params = [
                p for p, res in zip(self.params, result) if not res['last']
            ]
            if not self.req_ids:
                break

    def is_empty(self):
        return not self.req_ids

    def reset(self):
        self.data_collector = []
        self.rolling_batch = None
        # Store the results
        self.output_all = defaultdict(list)
        self.input_all = {}

        # Status variables, the remaining
        self.input_str = []
        self.params = []
        self.req_ids = []

        self.token_numbers = defaultdict(list)
