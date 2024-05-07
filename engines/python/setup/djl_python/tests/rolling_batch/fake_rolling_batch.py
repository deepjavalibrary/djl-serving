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
import random
from collections import OrderedDict
from transformers import AutoTokenizer
from djl_python.rolling_batch.rolling_batch import RollingBatch, stop_on_any_exception, Token


class FakeRollingBatch(RollingBatch):

    # TODO: Make properties is the only parameter, after refactoring all rolling batch handlers
    def __init__(self, model_id_or_path, properties, **kwargs):
        """
        Initializes the FakeRollingBatch.
        """
        super().__init__(**kwargs)
        self.sample_text = (
            "DJL-Serving is a powerful and user-friendly deep learning model serving solution "
            "that enables developers to easily deploy and serve their trained deep learning models."
            " With DJL-Serving, developers can quickly expose their models as web services or APIs,"
            " allowing them to integrate their deep learning models into various applications "
            "and systems seamlessly. The framework supports various deep learning frameworks like "
            "TensorFlow, PyTorch, MXNet, and more, making it versatile and adaptable to different model"
            " architectures. DJL-Serving is designed to be highly scalable and efficient, ensuring that"
            " models can handle high volumes of requests with low latency. Whether you are a researcher"
            " or a developer, DJL-Serving simplifies the process of serving deep learning models,"
            " enabling you to focus on creating innovative applications with ease."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                       padding_side="left",
                                                       trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokens = self.tokenizer.encode(self.sample_text)
        self.total_length = 32000
        while len(self.tokens) < self.total_length:
            self.tokens += self.tokens
        self.tokens = self.tokens[:self.total_length]
        self.cache = OrderedDict()

    def get_tokenizer(self):
        return self.tokenizer

    def reset(self):
        self.cache = OrderedDict()
        super().reset()

    @stop_on_any_exception
    def inference(self, input_data, parameters, adapters=None):
        batch_size = len(input_data)
        new_requests = self.get_new_requests(input_data, parameters,
                                             batch_size)

        for new_request in new_requests:
            max_len = new_request.parameters[
                "max_new_tokens"] if "max_new_tokens" in new_request.parameters else 256
            min_len = new_request.parameters[
                "min_new_tokens"] if "min_new_tokens" in new_request.parameters else 1
            max_len = max(min_len, max_len)
            max_len = random.randint(min_len, max_len)
            self.cache[new_request.id] = {
                "max_len": max_len,
                "cur_pos": -1,
                "finished": False
            }

        # fake inference
        for value in self.cache.values():
            value["cur_pos"] += 1
            if value["cur_pos"] == value["max_len"]:
                value["finished"] = True

        finished_id = []
        for (key, cache), request in zip(self.cache.items(),
                                         self.active_requests):
            # finish condition match
            if cache["finished"]:
                finished_id.append(key)
            token_id = self.tokens[cache["cur_pos"]]
            token_txt = " " + self.tokenizer.decode(token_id)
            request.set_next_token(Token(token_id, token_txt),
                                   cache["finished"])

        return self.postprocess_results()

    def preprocess_requests(self, requests):
        raise NotImplementedError("Not implemented for vLLM rolling batcher")


class FakeRollingBatchWithException(FakeRollingBatch):

    def __init__(self, model_id_or_path, properties, **kwargs):
        super().__init__(model_id_or_path, properties, **kwargs)
        self.dead_counter = 0
        self.dead_trigger = random.randint(1, 50)

    def reset(self):
        super().reset()
        self.dead_counter = 0
        self.dead_trigger = random.randint(1, 50)

    @stop_on_any_exception
    def inference(self, input_data, parameters, adapters=None):

        if self.dead_counter < self.dead_trigger:
            self.dead_counter += 1
            return super().inference(input_data, parameters, adapters)
        else:
            raise RuntimeError("Death trigger triggered...")
