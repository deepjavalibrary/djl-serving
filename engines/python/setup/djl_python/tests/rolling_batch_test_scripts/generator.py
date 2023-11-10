from collections import defaultdict

import os, sys

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
import torch.distributed as dist


def print_rank0(content):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(content)


# ===================== Runner Class =====================
class Generator:

    def __init__(self, rolling_batch):
        self.rolling_batch = rolling_batch

        # Store the results
        self.output_all = defaultdict(list)
        self.input_all = {}

        # Status variables, the remaining
        self.input_str = []
        self.params = []
        self.req_ids = []

    def reset(self):
        # Store the results
        self.output_all = defaultdict(list)
        self.input_all = {}

        # Status variables, the remaining
        self.input_str = []
        self.params = []
        self.req_ids = []

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

        for _ in range(step):
            result = self.rolling_batch.inference(self.input_str, self.params)
            for res, req_id in zip(result, self.req_ids):
                self.output_all[req_id].append(res['data'])
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
