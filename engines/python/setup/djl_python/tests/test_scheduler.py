import unittest
from djl_python.scheduler.lm_block import HuggingfaceGTP2Block
from djl_python.scheduler.seq_batch_scheduler_impl import GreedySeqBatchScheduler
from djl_python.scheduler.search_config import SearchConfig
import torch


class TestScheduler(unittest.TestCase):

    def test_lm_block(self):
        model_name = "gpt2"
        lm_block = HuggingfaceGTP2Block([model_name], {})

        input0 = [
            torch.tensor([[40, 2883, 6155, 351, 616, 13779, 3290]]),
            torch.arange(7)[None, :],
            torch.ones(7, dtype=torch.int64)[None, :]
        ]

        output0 = lm_block.forward(input0, None)

        # input with kv_cache
        past_key_values = output0[1]
        input_ids = torch.tensor([[404]])
        past_seq = past_key_values[0][0].shape[-2]
        position_ids = torch.tensor([[past_seq]])
        attention_mask = torch.ones(past_seq + 1, dtype=torch.int64)
        output1 = lm_block.forward([input_ids, position_ids, attention_mask],
                                   past_key_values)

    def test_greedy_scheduler_hf(self):
        model_name = "gpt2"
        lm_block = HuggingfaceGTP2Block([model_name], {})

        scheduler = GreedySeqBatchScheduler(lm_block, SearchConfig())

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
                                      220, 220, 220, 220, 220, 1858, 338, 257,
                                      640, 326, 314, 3505, 11, 618, 314, 750,
                                      407, 760
                                  ]])
        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids, kv_cache=None)
        for out in scheduler.increment_forward(2):
            pass

        # Load a kv_cache from file and test merging a shorter sequence
        input_ids = torch.tensor([[32]])
        request_ids = torch.tensor([[3]])
        kv_cache = torch.load(kv_cache_file)
        scheduler.add_request(input_ids, request_ids, kv_cache=kv_cache)

        # Test trim_and_collect
        for output in scheduler.increment_forward(100):
            pass

        results2 = scheduler.collect_results()
        assert len(results2) == 4


if __name__ == '__main__':
    unittest.main()
