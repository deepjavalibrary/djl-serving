import unittest
from djl_python.scheduler import HuggingfaceBlock
from djl_python.scheduler import GreedySeqBatchScheduler
from djl_python.scheduler import SearchConfig
from djl_python.scheduler.utils import compute_offsets, compute_position_ids, compute_attention_mask, merge_tensors, \
    trim_tensor
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
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        lm_block = HuggingfaceBlock(model)

        search_config = SearchConfig()
        search_config.max_seqlen = 30
        PAD = search_config.pad_token_id
        scheduler = GreedySeqBatchScheduler(lm_block, search_config)

        input_ids_0 = tokenizer.encode('Memories follow me left and right. I can', return_tensors='pt')
        request_ids = torch.tensor([[0]])

        # Save a kv_cache to file for later use
        kv_cache_file = "./kv_cache.pt"
        scheduler.init_forward(input_ids_0,
                               request_ids,
                               save_kv_cache_path=kv_cache_file)

        # Test add_request
        scheduler.add_request(request_ids, input_ids_0)

        input_ids_1 = tokenizer.encode("When your legs don't work like they used to before And I can't sweep you off",
                                       return_tensors='pt')
        input_ids_2 = torch.concat([torch.tensor([PAD, PAD, PAD, PAD, PAD]),
                                    tokenizer.encode("There's a time that I remember, when I did not know",
                                                     return_tensors='pt')[0]]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)

        # Test merging longer sequences
        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(request_ids, input_ids)
        for _ in scheduler.increment_forward(20):
            pass

        results = scheduler.results

        assert tokenizer.decode(torch.tensor(results[1][:30])) == "When your legs don't work like they used to before " \
                                                                  "And I can't sweep you off my feet, I can't do anything about it.\n"
        assert tokenizer.decode(torch.tensor(results[2][:30])) == "There's a time that I remember, when I did not " \
                                                                  "know what to do with my life. I was in a very bad mood. I was"
        assert tokenizer.decode(torch.tensor(results[0][:30])) == "Memories follow me left and right. I can't " \
                                                                  "remember the last time I saw a girl in a dress. I can't remember the last time"

        # Load a kv_cache from file and test merging a shorter sequence
        input_ids_1 = tokenizer.encode("When your legs don't work",
                                       return_tensors='pt')
        input_ids_2 = torch.concat([torch.tensor([PAD, PAD]),
                                    tokenizer.encode("There's a time",
                                                     return_tensors='pt')[0]]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)
        request_ids = torch.tensor([[3], [4]])

        # Load a kv_cache file to simulate a fixed reusable prefix which is pre-calculated
        kv_cache = torch.load(kv_cache_file)
        scheduler.add_request(request_ids, input_ids, kv_cache=kv_cache)

        # Test trim_and_collect
        for _ in scheduler.increment_forward(100):
            pass

        results = scheduler.results
        assert len(results) == 5
        assert tokenizer.decode(torch.tensor(results[3][:30])) == "!!!!!!!!!!When your legs don't work, you're going " \
                                                                  "to be a little bit more tired. I'm"
        assert tokenizer.decode(torch.tensor(results[4][:30])) == "!!!!!!!!!!There's a time and a place. I'm here to " \
                                                                  "stay. I'm here to stay."

    def test_seq_batcher(self):
        model_id = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_id)
        lm_block = HuggingfaceBlock(model)

        scheduler = GreedySeqBatchScheduler(lm_block, SearchConfig())

        # Initialize the SeqBatcher
        input_ids = torch.tensor(
            [[13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460]],
            dtype=torch.int64)
        request_ids = torch.tensor([[0]])
        seq_batcher = scheduler.init_forward(input_ids, request_ids)[0]

        input_ids_new = torch.tensor([
            [2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843, 314,
             460, 470, 16085, 345, 572
             ]])
        request_ids_new = torch.tensor([[1]])
        seq_batcher_new = scheduler.init_forward(input_ids_new, request_ids_new)[0]

        # Test SeqBatcher.add_batch
        seq_batcher.add_batch(seq_batcher_new)

        assert torch.all(seq_batcher.offsets == torch.tensor([[0], [8]]))
        assert torch.all(seq_batcher.request_uids == torch.tensor([[1], [0]]))
        assert seq_batcher.seq_len == 18
        assert len(seq_batcher.exit_index) == 0
        assert seq_batcher.batch_size == 2

        # Test collect_and_trim
        seq_batcher.exit_index.add(0)  # suppose the 0th sequence should exit
        seq_batcher.collect_and_trim()
        assert torch.all(seq_batcher.offsets == torch.tensor([[0]]))
        assert torch.all(seq_batcher.request_uids == torch.tensor([[0]]))
        assert seq_batcher.seq_len == 10
        assert len(seq_batcher.exit_index) == 0
        assert seq_batcher.batch_size == 1

    def test_utils(self):
        model_name = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name,
                                                  padding_side='left',
                                                  pad_token_id=220)

        input1 = [
            r'DeepMind Company is',
            r'Memories follow me left and right. I can'
        ]
        tokenizer.pad_token = '[PAD]'
        input_ids1 = tokenizer(input1, return_tensors='pt',
                               padding=True).input_ids

        config = SearchConfig()
        config.pad_token_id = 50256
        offsets = compute_offsets(input_ids1, config)
        assert torch.all(offsets == torch.tensor([[6], [0]]))

        attention_mask = compute_attention_mask(offsets, input_ids1.shape[-1])
        assert torch.all(attention_mask == torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

        position_ids = compute_position_ids(input_ids1.shape[0],
                                            input_ids1.shape[1],
                                            offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        assert torch.all(position_ids == torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))

        input2 = 'Fastertransformer is'

        input_ids2 = tokenizer(input2, return_tensors='pt',
                               padding=True).input_ids

        merged_tensor = merge_tensors(input_ids1,
                                      input_ids2,
                                      seq_delta=5)

        assert torch.all(merged_tensor == torch.tensor([
            [50256, 50256, 50256, 50256, 50256, 50256, 29744, 28478, 5834, 318],
            [13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460],
            [0, 0, 0, 0, 0, 37, 1603, 7645, 16354, 318]]))

        # removed the second row which has the largest sequence length
        trimmed_tensor = trim_tensor(merged_tensor,
                                     keep_indices=torch.tensor([0, 2]),
                                     trim_seq_len=5)
        assert torch.all(trimmed_tensor == torch.tensor(
            [[50256, 29744, 28478, 5834, 318], [37, 1603, 7645, 16354, 318]]))


if __name__ == '__main__':
    unittest.main()
