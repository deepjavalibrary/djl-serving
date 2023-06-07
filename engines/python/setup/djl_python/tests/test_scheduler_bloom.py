import unittest
from djl_python.scheduler import HuggingfaceBlock
from transformers import AutoConfig
import torch

from transformers import AutoTokenizer, BloomForCausalLM

from djl_python.scheduler import GreedySeqBatchScheduler
from djl_python.scheduler.lm_block import BloomBlock
from scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batch_scheduler_impl import ContrastiveSeqBatchScheduler


class TestSchedulerBloom(unittest.TestCase):

    def test_lm_block(self):
        model_id = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        encoding = tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids_0 = encoding.data['input_ids']
        seq_len = input_ids_0.shape[1]

        lm_block = HuggingfaceBlock(model)

        input0 = [
            torch.repeat_interleave(input_ids_0, dim=0, repeats=2),
            torch.repeat_interleave(torch.arange(seq_len)[None, :], dim=0, repeats=2),
            torch.repeat_interleave(torch.ones(seq_len, dtype=torch.int64)[None, :], dim=0, repeats=2)
        ]

        output0 = lm_block.forward(input0, None)

        model_config = AutoConfig.from_pretrained(model_id)
        assert len(output0[1]) == model_config.n_layer

        # input with kv_cache
        # k: [32, 64, 6], v: [32, 6, 64], [batch*head, kvDim, seq]
        past_key_values = output0[1]
        input_ids = torch.tensor([[404], [405]])
        past_seq = past_key_values[0][0].shape[-1]
        position_ids = torch.tensor([[past_seq], [past_seq]])
        attention_mask = torch.ones(2, past_seq + 1, dtype=torch.int64)
        output1 = lm_block.forward([input_ids, position_ids, attention_mask],
                                   past_key_values)
        assert len(output1[1]) == model_config.n_layer

    def test_contrastive_scheduler(self):
        model_id = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(model_id)
        model_config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lm_block = BloomBlock(model)

        search_config = SearchConfig()
        PAD = search_config.pad_token_id
        scheduler = ContrastiveSeqBatchScheduler(lm_block, search_config)

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
        scheduler.add_request(request_ids, input_ids, kv_cache=None)

        # Test merging longer sequences
        input_ids = torch.tensor([
            [
                2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843, 314,
                460, 470, 16085, 345, 572
            ],
            [
                PAD, PAD, PAD, PAD, PAD, 1858, 338, 257,
                640, 326, 314, 3505, 11, 618, 314, 750,
                407, 760
            ]])
        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(request_ids, input_ids, kv_cache=None)

        # Forward pass
        for _ in scheduler.increment_forward(20):
            pass

        results = scheduler.results

        assert torch.all(torch.tensor(results[1])[:30] == torch.tensor([2215, 534, 7405, 836, 470, 670, 588, 484, 973,
                                                                        284, 878, 843, 314, 460, 470, 16085, 345, 572,
                                                                        120, 9005, 651, 640, 2, 2003, 10463, 76506,
                                                                        4679, 39924, 23661, 229199]))

        assert torch.all(
            torch.tensor(results[2])[:30] == torch.tensor([1858, 338, 257, 640, 326, 314, 3505, 11, 618, 314,
                                                           750, 407, 760, 6333, 641, 1163, 3303, 23479, 4627, 6463,
                                                           1854, 62848, 3274, 12096, 30374, 15197, 2751, 10544, 21399,
                                                           25869]))

        assert torch.all(torch.tensor(results[0])[:30] == torch.tensor([13579, 1749, 1061, 502, 1364, 290, 826, 13, 314,
                                                                        460, 4041, 297, 55, 73, 15, 707, 40, 72,
                                                                        47, 77, 57, 74, 60, 189, 6, 4, 186976,
                                                                        33266, 22, 16783]))

        # Load a kv_cache from file
        input_ids = torch.tensor([[2215, 534, 7405, 836, 470, 670],
                                  [PAD, PAD, 1858, 338, 257, 640]])
        request_ids = torch.tensor([[3], [4]])

        # The kv_cache_file simulates a fixed resusable prefix whose kv_cache is pre-calculated
        kv_cache = torch.load(kv_cache_file)
        scheduler.add_request(request_ids, input_ids, kv_cache=kv_cache)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        assert torch.all(torch.tensor(results[3])[:30] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2215, 534,
                                                                        7405, 836, 470, 670, 460, 323, 189, 1020, 286,
                                                                        329, 72, 13, 76, 47, 78, 53, 74, 60]))

        assert torch.all(torch.tensor(results[4])[:30] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        1858, 338, 257, 640, 13, 89515, 8023, 46227,
                                                                        2766, 8023, 46227, 46227, 46227, 46227,
                                                                        46227, 46227, 46227,
                                                                        46227, 46227, 46227]))

        # print
        for i, ret in results.items():
            print('\n{}:'.format(i), tokenizer.decode(ret))


if __name__ == '__main__':
    unittest.main()
