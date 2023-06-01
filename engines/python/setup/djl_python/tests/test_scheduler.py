import unittest
from djl_python.scheduler import HuggingfaceBlock
from djl_python.scheduler import GreedySeqBatchScheduler
from djl_python.scheduler import SearchConfig
from djl_python.scheduler.utils import compute_offsets, compute_position_ids, compute_attention_mask, merge_tensors, trim_tensor
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
import torch


class TestScheduler(unittest.TestCase):

    def test_lm_block(self):
        model_id = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_id)
        lm_block = HuggingfaceBlock(model)

        input0 = [
            torch.tensor([[40, 2883, 6155, 351, 616, 13779, 3290]]),
            torch.arange(7)[None, :],
            torch.ones(7, dtype=torch.int64)[None, :]
        ]

        output0 = lm_block.forward(input0, None)

        model_config = AutoConfig.from_pretrained(model_id)
        assert len(output0[1]) == model_config.n_layer

        # input with kv_cache
        past_key_values = output0[1]
        input_ids = torch.tensor([[404]])
        past_seq = past_key_values[0][0].shape[-2]
        position_ids = torch.tensor([[past_seq]])
        attention_mask = torch.ones(past_seq + 1, dtype=torch.int64)
        output1 = lm_block.forward([input_ids, position_ids, attention_mask],
                                   past_key_values)
        assert len(output1[1]) == model_config.n_layer

    def test_greedy_scheduler(self):
        model_id = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_id)
        lm_block = HuggingfaceBlock(model)

        search_config = SearchConfig()
        PAD = search_config.pad_token_id
        scheduler = GreedySeqBatchScheduler(lm_block, search_config)

        input_ids = torch.tensor(
            [[13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460]],
            dtype=torch.int64)
        request_ids = torch.tensor([[0]])

        # Test init_forward
        kv_cache_file = "./kv_cache.pt"
        batch = scheduler.init_forward(input_ids,
                                       request_ids,
                                       kv_cache=None,
                                       save_kv_cache_path=kv_cache_file)
        scheduler.add_request(input_ids, request_ids, kv_cache=None)

        # Test merging longer sequences
        input_ids = torch.tensor([[
            2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843, 314,
            460, 470, 16085, 345, 572
        ],
                                  [
                                      PAD, PAD, PAD, PAD, PAD, 1858, 338, 257,
                                      640, 326, 314, 3505, 11, 618, 314, 750,
                                      407, 760
                                  ]])
        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids, kv_cache=None)
        for out in scheduler.increment_forward(10):
            pass

        # Load a kv_cache from file and test merging a shorter sequence
        input_ids = torch.tensor([[2215, 534, 7405, 836, 470, 670],
                                  [PAD, PAD, 1858, 338, 257, 640]])
        request_ids = torch.tensor([[3], [4]])
        # Here the kv_cache_file simulates a fixed reusable prefix whose kv_cache is pre-calculated
        kv_cache = torch.load(kv_cache_file)
        scheduler.add_request(input_ids, request_ids, kv_cache=kv_cache)

        # Test trim_and_collect
        for output in scheduler.increment_forward(100):
            pass

        results = scheduler.collect_results()
        assert len(results) == 5

        assert torch.all(results[1] == torch.tensor([
            2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843, 314,
            460, 470, 16085, 345, 572, 616, 3625, 11, 314, 460, 470, 466, 1997,
            546, 340, 13, 198
        ]))

        assert torch.all(results[2] == torch.tensor([
            1858, 338, 257, 640, 326, 314, 3505, 11, 618, 314, 750, 407, 760,
            644, 284, 466, 351, 616, 1204, 13, 314, 373, 287, 257, 845, 2089,
            10038, 13, 314, 373
        ]))

        assert torch.all(results[0] == torch.tensor([
            13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460, 470, 3505,
            262, 938, 640, 314, 2497, 257, 2576, 287, 257, 6576, 13, 314, 460,
            470, 3505, 262, 938, 640
        ]))

        assert torch.all(results[3] == torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2215, 534, 7405, 836, 470, 670, 11,
            345, 821, 1016, 284, 307, 257, 1310, 1643, 517, 10032, 13, 314,
            1101
        ]))

        assert torch.all(results[4] == torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 220, 220, 1858, 338, 257, 640, 290, 257,
            1295, 13, 314, 1101, 994, 284, 2652, 13, 314, 1101, 994, 284, 2652,
            13
        ]))

    def test_utils(self):
        model_name = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name,
                                                  padding_side='left',
                                                  pad_token_id=220)

        input1 = [
            r'DeepMind Company is', r'Memories follow me left and right. I can'
        ]
        tokenizer.pad_token = '[PAD]'
        input_ids1 = tokenizer(input1, return_tensors='pt',
                               padding=True).input_ids

        config = SearchConfig()
        config.pad_token_id = 50256
        offsets = compute_offsets(input_ids1, config)
        assert torch.all(offsets == torch.tensor([[6], [0]]))

        attention_mask = compute_attention_mask(input_ids1, config)
        assert torch.all(attention_mask == torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

        position_ids = compute_position_ids(input_ids1.shape[0],
                                            input_ids1.shape[1],
                                            offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        assert torch.all(position_ids == torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))

        input2 = 'Fastertransformer is'

        input_ids2 = tokenizer(input2, return_tensors='pt',
                               padding=True).input_ids

        merged_tensor = merge_tensors(input_ids1,
                                      input_ids2,
                                      5,
                                      1,
                                      is_pad_token=True)

        assert torch.all(merged_tensor == torch.tensor([[
            50256, 50256, 50256, 50256, 50256, 50256, 29744, 28478, 5834, 318
        ], [
            13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460
        ], [50256, 50256, 50256, 50256, 50256, 37, 1603, 7645, 16354, 318]]))

        # removed the second row which has the largest sequence length
        trimed_tensor = trim_tensor(merged_tensor,
                                    keep_indices=torch.tensor([0, 2]),
                                    trim_seq_len=5)
        assert torch.all(trimed_tensor == torch.tensor(
            [[50256, 29744, 28478, 5834, 318], [37, 1603, 7645, 16354, 318]]))


if __name__ == '__main__':
    unittest.main()
