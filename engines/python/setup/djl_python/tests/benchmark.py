from djl_python.scheduler import HuggingfaceBlock, BloomBlock
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoConfig
import torch
from collections import defaultdict

from transformers import AutoTokenizer, BloomForCausalLM

from djl_python.scheduler import SearchConfig
from djl_python.scheduler.seq_batch_scheduler_impl import ContrastiveSeqBatchScheduler, GreedySeqBatchScheduler
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from typing import List

class TestKit:
    def __init__(self, scheduler: SeqBatchScheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer

    def process_request(self,
                        request_uids_list: List[int],
                        input: List[str],
                        search_configs: List[SearchConfig] = None):
        batch_size = len(input)
        input_ids = self.tokenizer(input, return_tensors='pt', padding=True).input_ids.view(batch_size, -1)
        request_uids = torch.tensor(request_uids_list).view(-1, 1)
        results = defaultdict(list)

        self.scheduler.add_request(request_uids, input_ids, search_configs)
        while not self.scheduler.is_empty():
            output_ids = self.scheduler.inference_call()

            # collect output
            for request_uid, output_id in zip(request_uids_list, output_ids):
                results[request_uid].append(output_id.item())

            # trim the sequence batcher
            self.scheduler.seq_batcher.collect_and_trim()

        for v in results.values():
            tokenizer.decode(v)


if __name__ == '__main__':

    input = [r"When your legs don't work like they used to before And I can't sweep you off",
             r"There's a time that I remember, when I did not know"]
    model_id = "gpt2"

    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = "[PAD]"

    lm_block = HuggingfaceBlock(model)
    config = SearchConfig()
    config.max_seqlen = 10
    config.max_seqlen += len(input[0])
    scheduler = GreedySeqBatchScheduler(lm_block, config)

    test_kit = TestKit(scheduler, tokenizer)

    # Send requests
    request_uids = [0, 1]

    test_kit.process_request(request_uids, input)

