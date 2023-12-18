import torch

import os, sys

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.10/dist-packages/lmi_dist")

# Flash attention imports
from flash_attn import flash_attn_varlen_func

# lmi_vllm imports
from lmi_vllm import cache_ops
from lmi_vllm import attention_ops as lmi_vllm_attention_ops
# from lmi_vllm import cache_ops  # from vllm._C import cache_ops

# vllm imports
from lmi_dist.utils.paged_attention import paged_attention, multi_query_cached_kv_attention


def test_one_layer(file):
    load = torch.load(file)
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    query = load['query']
    kv_cache = load['kv_cache']
    block_tables = load['block_tables']
    input_lengths = load['input_lengths']  # [7, 9, 7, 8]
    max_s = input_lengths.max().item()

    # Ground truth: single-query cached paged attention
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    # attn_out_sqa = single_query_cached_paged_attn(query, kv_cache, block_tables, input_lengths, max_s)

    attn_output = torch.empty_like(query)
    softmax_scale = 0.125
    _, num_key_value_heads, key_value_head_size, block_size = kv_cache[1].shape
    num_seqs_sqa, num_heads, head_size = query.shape
    num_key_value_groups = num_heads // num_key_value_heads
    kv_head_mapping = torch.arange(
        0, num_key_value_heads, dtype=torch.int32,
        device=query.device).repeat_interleave(num_key_value_groups)

    paged_attention(
        attn_output,
        query,
        kv_cache[0],
        kv_cache[1],
        # input_metadata
        num_key_value_heads,
        kv_head_mapping,
        softmax_scale,
        block_tables,
        input_lengths,
        max_s,
    )
    attn_out_sqa = attn_output

    kv_slots_mapping = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 224, 225, 226, 227, 228, 229, 230, 231, 232, 448,
        449, 450, 451, 452, 453, 454, 672, 673, 674, 675, 676, 677, 678, 679
    ],
                                    device=query.device,
                                    dtype=torch.int32)

    # Test mq cached flash_attn
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    attn_out_mqa = multi_query_cached_kv_attention(
        query,
        kv_cache[0],
        kv_cache[1],
        # metedata
        softmax_scale,
        kv_slots_mapping,
        input_lengths)

    assert (attn_out_sqa - attn_out_mqa).abs().max() < 5e-3

    max_diff = 0
    if not (attn_out_sqa - attn_out_mqa).abs().max() < 1e-4:
        watch = (attn_out_sqa - attn_out_mqa).abs().view(-1).topk(50)
        diff = attn_out_sqa - attn_out_mqa
        val, idx = torch.topk(diff.abs().view(-1), 50)
        print('\n')
        print(file)
        print((attn_out_sqa - attn_out_mqa).abs().max())
        print(diff.view(-1)[idx])
        max_diff = max(max_diff, (attn_out_sqa - attn_out_mqa).abs().max())

    print('\n', max_diff)
    return max_diff


def main():
    """
    torch.save({"query": query, "kv_cache": kv_cache, "block_tables": block_tables, "input_lengths": input_lengths}, "sqa_input_0.pt")
    """
    max_err = 0
    avg_err = 0
    for i in range(22):
        file = f'./data/sqa_input_{i}.pt'
        err = test_one_layer(file)
        max_err = max(max_err, err)
        avg_err += 1 / 22 * err
    print(f"max_err_all_files: {max_err}")
    print(f"avg_err_all_files: {avg_err}")


if __name__ == '__main__':
    main()
