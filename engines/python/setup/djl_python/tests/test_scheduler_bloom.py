import unittest

from djl_python.seq_scheduler.lm_block import BloomBlock, FalconBlock, HuggingfaceBlock
from djl_python.seq_scheduler.seq_batch_scheduler import SeqBatchScheduler
from transformers import AutoConfig, BloomForCausalLM, AutoTokenizer
from djl_python.seq_scheduler.search_config import SearchConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSchedulerBloom(unittest.TestCase):

    def test_lm_block(self):
        model_id = "bigscience/bloom-560m"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if global_device.type == "cuda" else "cpu")

        device = model.device
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        encoding = tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids_0 = encoding.data['input_ids']
        seq_len = input_ids_0.shape[1]

        lm_block = BloomBlock(model)

        input = [
            torch.repeat_interleave(input_ids_0, dim=0, repeats=2).to(device),
            torch.repeat_interleave(torch.arange(seq_len)[None, :],
                                    dim=0,
                                    repeats=2).to(device),
            torch.repeat_interleave(torch.ones(seq_len,
                                               dtype=torch.int64)[None, :],
                                    dim=0,
                                    repeats=2).to(device)
        ]

        output0 = lm_block.forward(*input, None)

        model_config = AutoConfig.from_pretrained(model_id)
        assert len(output0.past_key_values) == model_config.n_layer

        # input with kv_cache
        # k: [32, 64, 6], v: [32, 6, 64], [batch*head, kvDim, seq]
        past_key_values = output0.past_key_values
        input_ids = torch.tensor([[404], [405]]).to(device)
        past_seq = past_key_values[0][0].shape[-2]
        position_ids = torch.tensor([[past_seq], [past_seq]]).to(device)
        attention_mask = torch.ones(2, past_seq + 1,
                                    dtype=torch.int64).to(device)
        output1 = lm_block.forward(input_ids, position_ids, attention_mask,
                                   past_key_values)
        assert len(output1.past_key_values) == model_config.n_layer

    def test_contrastive_scheduler(self):
        model_id = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if global_device.type == "cuda" else "cpu")
        device = model.device
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  padding_side='left')

        lm_block = BloomBlock(model)

        search_config = SearchConfig()
        search_config.pad_token_id = tokenizer.pad_token_id
        PAD = search_config.pad_token_id
        scheduler = SeqBatchScheduler(lm_block, "contrastive", search_config)

        input_ids_0 = tokenizer.encode(
            'Memories follow me left and right. I can',
            return_tensors='pt').to(device)
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
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0).to(device)

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
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0).to(device)
        request_ids = torch.tensor([[3], [4]])

        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        # print
        for i, ret in results.items():
            print('\n{}:'.format(i), tokenizer.decode(ret))

    def test_greedy_scheduler_llama(self):
        model_name = "seanmor5/tiny-llama-test"
        tokenizer = AutoTokenizer.from_pretrained(
            "openlm-research/open_llama_3b", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if global_device.type == "cuda" else "cpu")
        device = model.device

        lm_block = HuggingfaceBlock(model)

        tokenizer.pad_token_id = 0
        search_config = SearchConfig(pad_token_id=tokenizer.pad_token_id)
        PAD = search_config.pad_token_id
        scheduler = SeqBatchScheduler(lm_block, "greedy", search_config)

        input_ids_0 = tokenizer.encode(
            'Memories follow me left and right. I can',
            return_tensors='pt').to(device)
        request_ids = torch.tensor([[0]])

        # Test init_forward
        scheduler.add_request(input_ids_0, request_ids)
        for _ in scheduler.increment_forward(20):
            pass

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
        input_ids = torch.concat([input_ids_1, input_ids_2], dim=0).to(device)

        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        results = scheduler.collect_results()

        assert tokenizer.decode(
            results[0][:30]
        ) == "<s> Memories follow me left and right. I can nobodyMuslim implants mount algorithms depends Posts semi dissent gospelSRassociIm requirerientotti regime Nem TED"
        assert tokenizer.decode(
            results[1][:30]
        ) == "<s> When your legs don't work like they used to before And I can't sweep you offimp shifted atoms raids......seud theat Furthermore locom"
        assert tokenizer.decode(
            results[2][:20]
        ) == "<s> There's a time that I remember, when I did not knowouthThose aquaticlace Stewart"

    def test_greedy_scheduler_llama2_gptq(self):
        if global_device.type == 'cpu':
            return
        model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=False,
                                                  padding_size='left',
                                                  use_fast=True)
        tokenizer.pad_token = "[PAD]"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            revision="main",
            device_map="auto" if global_device.type == "cuda" else "cpu")
        device = model.device

        lm_block = HuggingfaceBlock(model)

        search_config = SearchConfig(pad_token_id=tokenizer.pad_token_id)
        scheduler = SeqBatchScheduler(lm_block, "greedy", search_config)

        input_ids_0 = tokenizer.encode(
            'Memories follow me left and right. I can',
            return_tensors='pt').to(device)
        request_ids = torch.tensor([[0]])

        # Test init_forward
        scheduler.add_request(input_ids_0, request_ids)
        for _ in scheduler.increment_forward(20):
            pass

        # Merge longer sequences
        input12 = [
            r"When your legs don't work like they used to before And I can't sweep you off",
            r"There's a time that I remember, when I did not know"
        ]
        input_ids = tokenizer(input12, return_tensors='pt',
                              padding=True).input_ids.to(device)

        request_ids = torch.tensor([[1], [2]])
        scheduler.add_request(input_ids, request_ids)

        # Forward pass
        for _ in scheduler.increment_forward(100):
            pass

        results = scheduler.collect_results()
        # The generated text of TheBloke/Llama-2-7b-Chat-GPTQ has randomness even for greedy.


if __name__ == '__main__':
    unittest.main()
