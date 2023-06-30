import unittest

from djl_python.scheduler.lm_block import BloomBlock, FalconBlock
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from transformers import AutoConfig, BloomForCausalLM, AutoTokenizer
from djl_python.scheduler.search_config import SearchConfig
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM


class TestSchedulerBloom(unittest.TestCase):

    def test_lm_block(self):
        model_id = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        encoding = tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids_0 = encoding.data['input_ids']
        seq_len = input_ids_0.shape[1]

        lm_block = BloomBlock(model)

        input0 = [
            torch.repeat_interleave(input_ids_0, dim=0, repeats=2),
            torch.repeat_interleave(torch.arange(seq_len)[None, :],
                                    dim=0,
                                    repeats=2),
            torch.repeat_interleave(torch.ones(seq_len,
                                               dtype=torch.int64)[None, :],
                                    dim=0,
                                    repeats=2)
        ]

        output0 = lm_block.forward(*input0, None)

        model_config = AutoConfig.from_pretrained(model_id)
        assert len(output0.past_key_values) == model_config.n_layer

        # input with kv_cache
        # k: [32, 64, 6], v: [32, 6, 64], [batch*head, kvDim, seq]
        past_key_values = output0.past_key_values
        input_ids = torch.tensor([[404], [405]])
        past_seq = past_key_values[0][0].shape[-2]
        position_ids = torch.tensor([[past_seq], [past_seq]])
        attention_mask = torch.ones(2, past_seq + 1, dtype=torch.int64)
        output1 = lm_block.forward(input_ids, position_ids, attention_mask,
                                   past_key_values)
        assert len(output1.past_key_values) == model_config.n_layer

    def test_contrastive_scheduler(self):
        model_id = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(model_id)
        model_config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  padding_side='left')

        lm_block = BloomBlock(model)

        search_config = SearchConfig()
        search_config.pad_token_id = tokenizer.pad_token_id
        PAD = search_config.pad_token_id
        scheduler = SeqBatchScheduler(lm_block, "contrastive", search_config)

        input_ids_0 = tokenizer.encode(
            'Memories follow me left and right. I can', return_tensors='pt')
        request_ids = torch.tensor([[0]])

        # Test init_forward
        scheduler.add_request(input_ids_0, request_ids)

        # Merge longer sequences
        input_ids_1 = tokenizer.encode(
            "When your legs don't work like they used to before And I can't sweep you off",
            return_tensors='pt')
        input_ids_2 = torch.concat([
            torch.tensor([PAD, PAD, PAD, PAD]),
            tokenizer.encode(
                "There's a time that I remember, when I did not know",
                return_tensors='pt')[0]
        ]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)

        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(20):
            pass

        results = scheduler.results

        assert tokenizer.decode(
            results[1][:30]
        ) == "When your legs don't work like they used to before And I can't sweep you off my lap But if you're still here I'll take care of it If you're"
        assert tokenizer.decode(
            results[2][:20]
        ) == "There's a time that I remember, when I did not know what it was like to live in this"
        assert tokenizer.decode(
            results[0][:30]
        ) == "Memories follow me left and right. I can feel them moving around in my body, like they’re trying to tell me something about where I’m going"

        # Merge shorter sequences
        input_ids_1 = tokenizer.encode("When your legs don't work",
                                       return_tensors='pt')
        input_ids_2 = torch.concat([
            torch.tensor([PAD, PAD]),
            tokenizer.encode("There's a time", return_tensors='pt')[0]
        ]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)
        request_ids = torch.tensor([[3], [4]])

        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        # print
        for i, ret in results.items():
            print('\n{}:'.format(i), tokenizer.decode(ret))

    def test_contrastive_scheduler_falcon(self):
        model_name = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True)

        lm_block = FalconBlock(model)

        search_config = SearchConfig()
        PAD = search_config.pad_token_id
        scheduler = SeqBatchScheduler(lm_block, "contrastive", search_config)

        input_ids_0 = tokenizer.encode(
            'Memories follow me left and right. I can', return_tensors='pt')
        request_ids = torch.tensor([[0]])

        # Test init_forward
        scheduler.add_request(input_ids_0, request_ids)

        # Merge longer sequences
        input_ids_1 = tokenizer.encode(
            "When your legs don't work like they used to before And I can't sweep you off",
            return_tensors='pt')
        input_ids_2 = torch.concat([
            torch.tensor([PAD, PAD, PAD, PAD, PAD, PAD]),
            tokenizer.encode(
                "There's a time that I remember, when I did not know",
                return_tensors='pt')[0]
        ]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)

        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(20):
            pass

        results = scheduler.results

        # assert tokenizer.decode(
        #     results[1][:30]
        # ) == "When your legs don't work like they used to before And I can't sweep you off my lap But if you're still here I'll take care of it If you're"
        # assert tokenizer.decode(
        #     results[2][:20]
        # ) == "There's a time that I remember, when I did not know what it was like to live in this"
        # assert tokenizer.decode(
        #     results[0][:30]
        # ) == "Memories follow me left and right. I can feel them moving around in my body, like they’re trying to tell me something about where I’m going"

        # Merge shorter sequences
        input_ids_1 = tokenizer.encode("When your legs don't work",
                                       return_tensors='pt')
        input_ids_2 = torch.concat([
            torch.tensor([PAD, PAD]),
            tokenizer.encode("There's a time", return_tensors='pt')[0]
        ]).view(1, -1)
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0)
        request_ids = torch.tensor([[3], [4]])

        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        # print
        for i, ret in results.items():
            print('\n{}:'.format(i), tokenizer.decode(ret))



if __name__ == '__main__':
    unittest.main()
