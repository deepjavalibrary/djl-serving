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

from functools import wraps
import time
import argparse


def timeit(repetitions=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0.0
            for _ in range(repetitions):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time += end_time - start_time
            avg_time = total_time / repetitions
            print(f'Function: {func.__name__}\nAverage time for {repetitions} repetitions: {avg_time:.4f} seconds')
            return avg_time
        return wrapper
    return decorator


class TestKit:
    def __init__(self, scheduler: SeqBatchScheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer

    @torch.no_grad()
    def process_request(self,
                        request_uids: torch.tensor,
                        input: List[str],
                        search_configs: List[SearchConfig] = None):
        batch_size = len(input)
        input_ids = self.tokenizer(input, return_tensors='pt', padding=True).input_ids.view(batch_size, -1)

        results = self.pure_inference(request_uids, input_ids, search_configs)
        for v in results.values():
            self.tokenizer.decode(v)

    def pure_inference(self, request_uids, input_ids, search_configs):
        results = defaultdict(list)

        self.scheduler.add_request(request_uids, input_ids, search_configs)
        while not self.scheduler.is_empty():
            output_ids = self.scheduler.inference_call()

            # collect output
            for request_uid, output_id in zip(request_uids, output_ids):
                results[request_uid.item()].append(output_id.item())

            # trim the sequence batcher
            self.scheduler.seq_batcher.collect_and_trim()

        return results


def get_model(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_id == "bloom560":
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        model = model.to(device)
        lm_block = BloomBlock(model)

    elif model_id == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_id)
        model = model.to(device)
        lm_block = HuggingfaceBlock(model)

    return lm_block

def main(args):
    input = [r"When your legs don't work like they used to before And I can't sweep you off",
             r"There's a time that I remember, when I did not know"]

    model_id = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = "[PAD]"

    # search config
    config = SearchConfig()
    config.pad_token_id = tokenizer.pad_token_id
    config.max_seqlen = args.max_seq_len + max(len(input[0]), len(input[1]))
    config.max_seqlen += len(input[0])
    scheduler = GreedySeqBatchScheduler(get_model(model_id), config)

    # Init test kit
    test_kit = TestKit(scheduler, tokenizer)

    # Send requests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    request_uids = torch.tensor([0, 1], device=device).view(-1, 1)

    reps = args.reps

    @timeit(reps)
    def test_run(test_kit, request_uid, input):
        test_kit.process_request(request_uids, input)

    avg_time = test_run(test_kit, request_uids, input)
    print("avg_time: ", avg_time)

    @timeit(3)
    def display(x):
        time.sleep(.2)

    display("foo")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['gpt2', 'bloom560'], default="gpt2")
    args = parser.parse_args()
    main(args)