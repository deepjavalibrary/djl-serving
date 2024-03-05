from functools import wraps, partial
import time
import numpy as np
import scipy.stats as stats
import subprocess as sp
import torch
import shutil
from typing import List

import os

script_directory = os.path.dirname(os.path.abspath(__file__))

from seq_scheduler.lm_block import HuggingfaceBlock, BloomBlock

from transformers import AutoTokenizer, BloomForCausalLM, AutoModelForCausalLM


def parse_input(default, varargin):
    for attr, value in varargin.__dict__.items():
        setattr(default, attr, value)


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


def timeit(func=None, *, repetitions=5):
    if func is None:
        return partial(timeit, repetitions=repetitions)

    @wraps(func)
    def wrapper(*args, **kwargs):
        total_time, total_mem = 0.0, 0.0
        annealing = 0
        data, data_m, data_m2, data_accp = [], [], [], []
        last_output = None
        for idx in range(repetitions + annealing):
            start_time = time.perf_counter()
            last_output = func(*args, **kwargs)
            end_time = time.perf_counter()
            if idx >= annealing:
                total_time += end_time - start_time
                data.append(end_time - start_time)
                data_m.append(last_output[3])
                data_m2.append(last_output[4])
                data_accp.append(
                    np.mean([
                        e for sublist in last_output[1].values()
                        for e in sublist if e > 0
                    ]))

        avg_time = total_time / repetitions
        data_time = np.array(data)
        data_mem = np.array(data_m)
        data_mem2 = np.array(data_m2)
        data_accp = np.array(data_accp)
        print(
            f'Function: {func.__name__}\nAverage time for {repetitions} repetitions: {avg_time:.4f} sec / rep'
        )

        # Output results:
        batch_size = len(args[-1])
        max_gen_len = 0
        if 'scheduler' in args[0].__dict__:
            max_gen_len = args[
                0].scheduler.default_search_config.max_new_seqlen
        elif 'param' in args[0].__dict__:
            max_gen_len = args[0].param['max_new_tokens']

        seq_thru_put_data = batch_size / data_time  # req/sec
        token_latency_data = 1000 * data_time / (batch_size * max_gen_len
                                                 )  # sec/token
        return avg_time, batch_size * max_gen_len, stat_tool(
            seq_thru_put_data), stat_tool(token_latency_data), stat_tool(
                data_mem), stat_tool(data_mem2), last_output, stat_tool(
                    data_accp)

    return wrapper


def get_gpu_memory():
    if shutil.which("nvidia-smi") is None:
        return [0]
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]


class PeakMemory:

    def __init__(self) -> None:
        info = torch.cuda.mem_get_info()
        self.peak_memory = info[1] - info[0]

    def reset(self):
        info = torch.cuda.mem_get_info()
        self.peak_memory = info[1] - info[0]

    def aggregate(self):
        info = torch.cuda.mem_get_info()
        self.peak_memory = max(self.peak_memory, info[1] - info[0])

    def get(self):
        return self.peak_memory


def get_model_tokenizer(model_id, flash_attn=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    lm_block, tokenizer, model_id_or_path = None, None, None
    if model_id == "bloom560":
        model_id_or_path = "bigscience/bloom-560m"
        model = BloomForCausalLM.from_pretrained(model_id_or_path)
        model = model.to(device)
        lm_block = BloomBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    elif model_id == "gpt2":
        model_id_or_path = 'gpt2'
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)
        lm_block = HuggingfaceBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    elif model_id == "gpt_neox":
        from lmi_dist.models.gpt_neox import GPTNeoxSharded
        from lmi_dist.utils import download_and_convert_weights

        model_id_or_path = "EleutherAI/gpt-neox-20b"
        download_and_convert_weights(model_id_or_path)
        model = GPTNeoxSharded(model_id_or_path)

        lm_block = HuggingfaceBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    elif model_id == "llama":
        model_id_or_path = "huggyllama/llama-7b"
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map='auto' if device.type == 'cuda' else 'cpu',
            use_flash_attention_2=flash_attn,
            torch_dtype=torch.float16)
        lm_block = HuggingfaceBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    elif model_id == "llama2":
        model_id_or_path = "TheBloke/Llama-2-7B-Chat-fp16"
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map='auto' if device.type == 'cuda' else 'cpu',
            use_flash_attention_2=flash_attn,
            torch_dtype=torch.float16)
        lm_block = HuggingfaceBlock(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path,
                                                  padding_side='left')
        tokenizer.pad_token = "[PAD]"

    return lm_block, tokenizer, model_id_or_path


def get_input_str(input_size) -> List[str]:
    input = [r"The new movie that got Oscar this year"]
    for prompt_id in range(input_size - 1):
        file_dir = script_directory + "/../resources/"
        file_name = "prompt" + str(prompt_id) + ".csv"
        with open(file_dir + file_name, "r") as file:
            prompt_str = file.read()
            input.append(prompt_str)
    return input


def sparsity(input_token_len: List[int]):
    input_token_len = sorted(input_token_len, reverse=True)
    max_len = input_token_len[0]
    return sum(max_len - x
               for x in input_token_len) / (max_len * len(input_token_len))
