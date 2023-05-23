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

import torch

from djl_python.scheduler.search_config import SearchConfig

PAD_TOKEN_ID = 220


def merge_tensors(tensor1: torch.Tensor,
                  tensor2: torch.Tensor,
                  seq_delta,
                  seq_order,
                  is_pad_token=False) -> torch.Tensor:
    if seq_delta == 0:
        return torch.cat([tensor1, tensor2], dim=0)

    shape1 = tensor1.shape
    shape2 = tensor2.shape

    delta_shape = list(shape1)
    delta_shape[0] = shape2[0]

    if is_pad_token:
        delta_tensor = torch.full(delta_shape,
                                  fill_value=PAD_TOKEN_ID,
                                  dtype=tensor1.dtype)
    else:
        delta_tensor = torch.zeros(delta_shape, dtype=tensor1.dtype)

    # augment the batch 1
    tensor1 = torch.cat([tensor1, delta_tensor], dim=0)

    if seq_order == 1:
        tensor1[shape1[0]:, seq_delta:, ...] = tensor2
    elif seq_order == 2:
        tensor1[shape1[0]:, :, seq_delta:, ...] = tensor2
    elif seq_order == -1:
        tensor1[shape1[0]:] = tensor2

    return tensor1


def trim_tensor(tensor: torch.Tensor,
                keep_indices: torch.Tensor,
                trim_seq_len: int,
                seq_order=1) -> torch.Tensor:
    if trim_seq_len == 0:
        return tensor[keep_indices]

    if seq_order == 1:
        return tensor[keep_indices, trim_seq_len:, ...]
    elif seq_order == 2:
        return tensor[keep_indices, :, trim_seq_len:, ...]
    elif seq_order == -1:
        return tensor[keep_indices]


def compute_offsets(input_ids: torch.Tensor,
                    config: SearchConfig) -> torch.Tensor:
    num_batch = input_ids.shape[0]
    seq_size = input_ids.shape[1]

    offsets = []
    for i in range(num_batch):
        sequence = input_ids[i].tolist()
        index = 0
        while index < seq_size:
            if sequence[index] != config.pad_token_id:
                break
            index += 1

        offsets.append(index)

    return torch.tensor(offsets, dtype=torch.int64).view(-1, 1)


def compute_attention_mask(input_ids, config: SearchConfig):
    num_batch = input_ids.shape[0]
    seq_size = input_ids.shape[1]

    # attention_mask
    attention_mask = torch.repeat_interleave(torch.ones(
        [1, input_ids.shape[-1]], dtype=torch.int64).reshape(1, -1),
                                             dim=0,
                                             repeats=num_batch)

    # Linear searches the offset and set the mask
    for i in range(num_batch):
        sequence = input_ids[i].tolist()
        index = 0
        while index < seq_size:
            if sequence[index] != config.pad_token_id:
                break
            index += 1

        attention_mask[i][0:index] = 0

    return attention_mask


def compute_position_ids(input_ids: torch.Tensor, offsets: torch.Tensor,
                         past_seq_len: int, repeat: int):
    position_range = torch.arange(start=past_seq_len,
                                  end=past_seq_len + input_ids.shape[-1],
                                  step=1,
                                  dtype=torch.int64).view(1, -1)

    position_ids = torch.repeat_interleave(position_range,
                                           dim=0,
                                           repeats=input_ids.shape[0])

    position_ids_shifted = position_ids - torch.repeat_interleave(
        offsets.view(-1, 1), dim=0, repeats=repeat)

    position_ids = torch.maximum(position_ids_shifted,
                                 torch.zeros_like(position_ids_shifted))
    return position_ids
