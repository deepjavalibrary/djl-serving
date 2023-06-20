from djl_python.scheduler import HuggingfaceBlock, BloomBlock
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from collections import defaultdict

from transformers import AutoTokenizer, BloomForCausalLM

from djl_python.scheduler import SearchConfig
from djl_python.scheduler.seq_batcher_impl import GreedySeqBatcher, ContrastiveSeqBatcher
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from typing import List

from functools import wraps
import time
import argparse

import numpy as np
import scipy.stats as stats


class TestKit:

    def __init__(self, scheduler: SeqBatchScheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer

    @torch.no_grad()
    def pure_inference(self,
                       request_uids: torch.tensor,
                       input_ids: torch.tensor,
                       search_configs: List[SearchConfig] = None):
        results = defaultdict(list)

        self.scheduler.add_request(input_ids,
                                   request_uids,
                                   search_configs=search_configs)

        while not self.scheduler.is_empty():
            output_ids, request_uids, _ = self.scheduler.inference_call()

            # collect output
            for request_uid, output_id in zip(request_uids, output_ids):
                results[request_uid].extend(output_id)

        return results


def get_model_tokenizer(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_block, tokenizer = None, None
    if model_id == "bloom560":
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        model = model.to(device)
        lm_block = BloomBlock(model)
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m",
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    elif model_id == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_id)
        model = model.to(device)
        lm_block = HuggingfaceBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    return lm_block, tokenizer


def stat_tool(experiment_results):
    # Compute the average
    average = np.mean(experiment_results)

    # Compute the standard error using the standard deviation and sample size
    standard_deviation = np.std(experiment_results)
    sample_size = len(experiment_results)
    standard_error = standard_deviation / np.sqrt(sample_size)

    # Compute the confidence interval using a t-distribution
    confidence_level = 0.95  # 95% confidence level
    degrees_of_freedom = sample_size - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error = t_value * standard_error
    confidence_interval = (average - margin_of_error,
                           average + margin_of_error)

    stat_result = {
        "avg":
        average if ~np.isnan(average) else -1,
        "std":
        standard_error if ~np.isnan(standard_error) else -1,
        "conf_intv":
        confidence_interval
        if ~np.any(np.isnan(np.array(confidence_interval))) else [-1, -1]
    }
    return stat_result


def timeit(repetitions=5):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0.0
            annealing = 1
            data = []
            for idx in range(repetitions + annealing):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                if idx >= annealing:
                    total_time += end_time - start_time
                    data.append(end_time - start_time)
            avg_time = total_time / repetitions
            data_time = np.array(data)
            print(
                f'Function: {func.__name__}\nAverage time for {repetitions} repetitions: {avg_time:.4f} seconds'
            )

            # Output results:
            if len(args) == 3:
                batch_size, init_seq_len = args[2].shape
                max_gen_len = args[0].scheduler.default_search_configs[
                    "$"].max_new_seqlen
                seq_thru_put_data = batch_size / data_time  # req/sec
                token_latency_data = 1000 * data_time / (
                    batch_size * max_gen_len)  # sec/token
                return avg_time, batch_size * max_gen_len, stat_tool(
                    seq_thru_put_data), stat_tool(token_latency_data),
            else:
                return None

        return wrapper

    return decorator


def main(args):
    model_id = args.model

    model, tokenizer = get_model_tokenizer(model_id)

    ## Test homogeneous request
    input = [r"The new movie that got Oscar this year"]
    input = input * args.concurrency
    search_algo = {
        "greedy": GreedySeqBatcher,
        "contrastive": ContrastiveSeqBatcher
    }
    seq_batcher_cls = search_algo[args.batch_type]
    batch_size = len(input)

    # Prepare requests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    request_uids = torch.tensor(range(batch_size), device=device).view(-1, 1)
    input_ids = tokenizer(input, return_tensors='pt', padding=True) \
        .input_ids.view(batch_size, -1)
    input_ids = input_ids.to(device)

    # search config
    config = SearchConfig()
    config.pad_token_id = tokenizer.pad_token_id
    config.max_new_seqlen = args.max_gen_len
    scheduler = SeqBatchScheduler(model, seq_batcher_cls, config)

    # Init test kit
    test_kit = TestKit(scheduler, tokenizer)

    reps = args.reps

    @timeit(reps)
    def test_run(test_kit, request_uids, input_ids):
        test_kit.pure_inference(request_uids, input_ids)

    avg_time, tokens, seq_thru_put_stat, token_latency_stat = test_run(
        test_kit, request_uids, input_ids)
    print(
        f"avg_time: {avg_time}, "
        f"tot_tokens: {tokens}, "
        f"seq_thru_put: {seq_thru_put_stat['avg']:.3g} reqs/sec, \n"
        f"\t err: {seq_thru_put_stat['std']:.3g}, \n"
        f"\t conf_intv: {seq_thru_put_stat['conf_intv'][0]:.3g}, {seq_thru_put_stat['conf_intv'][1]:.3g} \n"
        f"token_latency: {token_latency_stat['avg']:.3g} ms/token \n"
        f"\t err: {token_latency_stat['std']:.3g}, \n"
        f"\t conf_intv: {token_latency_stat['conf_intv'][0]:.3g}, {token_latency_stat['conf_intv'][1]:.3g} \n"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('-r', '--reps', dest='reps', type=int, default=2)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=2)
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560'],
                        default="bloom560")
    parser.add_argument('--batch_type',
                        type=str,
                        choices=['greedy', 'contrastive'],
                        default="greedy")
    args = parser.parse_args()
    for c in {1, 2, 4}:
        args.concurrency = c
        main(args)
