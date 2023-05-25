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
from typing import Tuple

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


def nudge_tensor(tensor: torch.Tensor,
                 offsets: torch.Tensor,
                 init_seq_len: int,
                 seq_order: int):
    if len(offsets.shape) < 2:
        offsets = offsets.view(-1, 1)

    if torch.all(offsets == 0) or init_seq_len == 0:
        return tensor

    tensor_new = tensor.clone()
    for i in range(offsets.shape[0]):
        offset = offsets[i].item()
        if seq_order == 1:
            tensor_new[i, offset: offset + init_seq_len, ...] = tensor[i, :init_seq_len, ...]
            tensor_new[i, :offset, ...] = 0
        elif seq_order == 2:
            tensor_new[i, :, offset: offset + init_seq_len, ...] = tensor[i, :, :init_seq_len, ...]

    return tensor_new


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


def compute_position_ids(batch_size: int, input_seq_len: int, offsets: torch.Tensor, past_seq_len: int,
                         repeat_offset: int):
    # past_seq_len: the starting position of the whole batch
    # repeat_offset: interleave_repeat the offsets to match the batch size of input_ids
    position_range = torch.arange(start=past_seq_len,
                                  end=past_seq_len + input_seq_len,
                                  step=1,
                                  dtype=torch.int64).view(1, -1)

    position_ids = torch.repeat_interleave(position_range,
                                           dim=0,
                                           repeats=batch_size)

    position_ids_shifted = position_ids - torch.repeat_interleave(
        offsets.view(-1, 1), dim=0, repeats=repeat_offset)

    position_ids = torch.maximum(position_ids_shifted,
                                 torch.zeros_like(position_ids_shifted))
    return position_ids


def assemble_prefix_kv_cache(input_ids, position_ids, attention_mask, kv_cache: Tuple):
    if kv_cache is None:
        return None, position_ids, attention_mask, None

    if kv_cache[0][0].shape[0] > 1:
        raise Exception(
            "When kv_cache that precedes the input_ids is provided, the init_forward is restricted"
            " to process one sequence at a time, which is not padded. This avoids the padding "
            "bubble between the precedent kv_cache and the input_ids.")

    init_kv_cache_len = kv_cache[0][0].shape[2]
    batch_size = input_ids.shape[0]

    attention_mask = torch.cat([
        torch.ones(
            (batch_size, init_kv_cache_len), dtype=torch.int64), attention_mask
    ], dim=1)
    position_ids += init_kv_cache_len
    dummy_input_ids = torch.full([batch_size, init_kv_cache_len],
                                 fill_value=0,
                                 dtype=input_ids.dtype)

    kv_cache_copied = []
    for k, v in kv_cache:
        k_copied = torch.repeat_interleave(k,
                                           dim=0,
                                           repeats=batch_size)
        v_copied = torch.repeat_interleave(v,
                                           dim=0,
                                           repeats=batch_size)
        kv_cache_copied.append((k_copied, v_copied))
    kv_cache = tuple(kv_cache_copied)

    return dummy_input_ids, position_ids, attention_mask, kv_cache
