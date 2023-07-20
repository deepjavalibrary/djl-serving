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
from typing import Tuple, Dict, List, Union

import torch

from djl_python.scheduler.lm_block import LMBlock
from djl_python.scheduler.search_config import SearchConfig


def merge_tensors(tensor1: torch.Tensor,
                  tensor2: torch.Tensor,
                  seq_delta,
                  seq_order=1) -> torch.Tensor:
    if seq_delta == 0:
        return torch.cat([tensor1, tensor2], dim=0)

    shape1 = tensor1.shape
    shape2 = tensor2.shape

    delta_shape = list(shape1)
    delta_shape[0] = shape2[0]

    delta_tensor = torch.zeros(delta_shape,
                               dtype=tensor1.dtype,
                               device=tensor1.device)
    assert tensor1.device == tensor2.device

    # Augment the batch 1
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
                seq_order: int = 1) -> torch.Tensor:
    if trim_seq_len == 0:
        return tensor[keep_indices.to(tensor.device)]

    if seq_order == 1:
        return tensor[keep_indices, trim_seq_len:, ...]
    elif seq_order == 2:
        return tensor[keep_indices.to(tensor.device), :, trim_seq_len:, ...]
    elif seq_order == -1:
        return tensor[keep_indices]


def nudge_tensor(tensor: torch.Tensor, offsets: torch.Tensor,
                 init_kv_cache_len: int, seq_order: int):
    """
    This is used with a prefix kv_cache input. The init_kv_cache_len part of the tensor is nudged towards right,
    by the displacement specified in offsets, so as to squeeze the padding bubble.
    """
    if len(offsets.shape) < 2:
        offsets = offsets.view(-1, 1)

    if torch.all(offsets == 0) or init_kv_cache_len == 0:
        return tensor

    tensor_new = tensor.clone()
    offsets_list = offsets.view(-1).tolist()
    for i in range(len(offsets_list)):
        offset = offsets_list[i]
        if seq_order == 1:
            tensor_new[i, offset:offset + init_kv_cache_len,
                       ...] = tensor[i, :init_kv_cache_len, ...]
            tensor_new[i, :offset, ...] = 0
        elif seq_order == 2:
            tensor_new[i, :, offset:offset + init_kv_cache_len,
                       ...] = tensor[i, :, :init_kv_cache_len, ...]

    return tensor_new


def compute_offsets(input_ids: torch.Tensor,
                    pad_token_ids: List[int]) -> torch.Tensor:
    num_batch = input_ids.shape[0]
    seq_size = input_ids.shape[1]

    offsets = []
    for sequence, pad_token_id in zip(input_ids.tolist(), pad_token_ids):
        index = 0
        while index < seq_size:
            if sequence[index] != pad_token_id:
                break
            index += 1

        offsets.append(index)

    return torch.tensor(offsets, dtype=torch.int64,
                        device=input_ids.device).view(-1, 1)


def compute_position_ids(batch_size: int, input_seq_len: int,
                         offsets: torch.Tensor, past_seq_len: int,
                         repeat_offset: int):
    # past_seq_len: the starting position of the whole batch
    # repeat_offset: interleave_repeat the offsets to match the batch size of input_ids
    position_range = torch.arange(start=past_seq_len,
                                  end=past_seq_len + input_seq_len,
                                  step=1,
                                  dtype=torch.int64,
                                  device=offsets.device).view(1, -1)

    position_ids = torch.repeat_interleave(position_range,
                                           dim=0,
                                           repeats=batch_size)

    position_ids_shifted = position_ids - torch.repeat_interleave(
        offsets.view(-1, 1), dim=0, repeats=repeat_offset)

    position_ids = torch.maximum(
        position_ids_shifted,
        torch.zeros_like(position_ids_shifted,
                         device=position_ids_shifted.device))
    return position_ids


def compute_attention_mask(offsets: torch.tensor,
                           seq_len: int,
                           repeat_offset: int = 1):
    if len(offsets.shape) != 2:
        raise Exception("wrong shape of offsets")

    batch_size = len(offsets) * repeat_offset
    past_attention_mask = torch.ones(batch_size,
                                     seq_len,
                                     dtype=torch.int64,
                                     device=offsets.device)
    for i, offset in enumerate(offsets):
        repeat_part = slice(i * repeat_offset, (i + 1) * repeat_offset)
        past_attention_mask[repeat_part, :offset] = 0

    return past_attention_mask


def assemble_prefix_kv_cache(input_ids, position_ids, attention_mask,
                             kv_cache: Tuple, kv_cache_input_ids):
    """
    This is used with a prefix kv cache input, to infer the correct position_ids and attention_mask.
    """
    if kv_cache is None:
        return None, position_ids, attention_mask, None

    if kv_cache[0][0].shape[0] > 1:
        raise Exception(
            "When kv_cache as a fixed prefix to the input_ids is provided, the init_forward is restricted"
            " to apply this common kv_cache to all inputs. If you have different kv_cache, do it in a "
            "different add_request() call")

    init_kv_cache_len = kv_cache[0][0].shape[2]
    batch_size = input_ids.shape[0]

    attention_mask = torch.cat([
        torch.ones((batch_size, init_kv_cache_len),
                   dtype=torch.int64,
                   device=attention_mask.device), attention_mask
    ],
                               dim=1)
    position_ids += init_kv_cache_len
    # If in the future not only prefix kv_cache is given, but also prefix token ids are given,
    # then the dummy_token_ids will still be used and only assemble the prefix at the final output.
    dummy_input_ids = torch.full(
        [batch_size, init_kv_cache_len],
        fill_value=0,
        dtype=input_ids.dtype,
        device=input_ids.device
    ) if kv_cache_input_ids is None else torch.repeat_interleave(
        kv_cache_input_ids, dim=0, repeats=batch_size)

    kv_cache_copied = []
    for k, v in kv_cache:
        k_copied = torch.repeat_interleave(k, dim=0, repeats=batch_size)
        v_copied = torch.repeat_interleave(v, dim=0, repeats=batch_size)
        kv_cache_copied.append((k_copied, v_copied))
    kv_cache = tuple(kv_cache_copied)

    return dummy_input_ids, position_ids, attention_mask, kv_cache


def compute_kv_cache(input_ids: torch.Tensor,
                     lm_block: LMBlock,
                     save_kv_cache_paths: List[str] = None,
                     search_configs: Union[List[SearchConfig], None] = None):
    if save_kv_cache_paths and input_ids.shape[0] != len(save_kv_cache_paths):
        raise Exception(
            "input_ids.shape does not match save_kv_cache_paths shape or is illegal"
        )

    pad_token_ids = []
    if not search_configs:
        for token_ids in input_ids:
            first_token_id = token_ids[0].item()
            pad_token_id = (first_token_id - 1) if first_token_id != 0 else 0
            pad_token_ids.append(pad_token_id)
    else:
        pad_token_ids.append(search_config.pad_token_id
                             for search_config in search_configs)

    initial_offsets = compute_offsets(input_ids, pad_token_ids)
    attention_mask = compute_attention_mask(initial_offsets,
                                            input_ids.shape[-1])
    position_ids = compute_position_ids(input_ids.shape[0],
                                        input_ids.shape[1],
                                        initial_offsets,
                                        past_seq_len=0,
                                        repeat_offset=1)

    # Forward call
    model_input = [input_ids, position_ids, attention_mask]
    lm_output = lm_block.forward(*model_input, past_key_values=None)
    past_key_values = lm_output.past_key_values

    # Save kv_cache of input_ids
    last_kv_cache = None
    for idx in range(initial_offsets.numel()):
        kv_cache_list = []
        for k, v in past_key_values:
            offset = initial_offsets[idx].item()
            k_idx = k[idx, :, offset:, :].unsqueeze(dim=0)
            v_idx = v[idx, :, offset:, :].unsqueeze(dim=0)
            kv_cache_list.append((k_idx, v_idx))

        last_kv_cache = tuple(kv_cache_list)
        if save_kv_cache_paths:
            torch.save(last_kv_cache, save_kv_cache_paths[idx])

    return last_kv_cache
