#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import functools
import tempfile
from typing import Dict, Tuple, Callable

from transformers import (BloomConfig, LlamaConfig, GPT2Config, GPTNeoXConfig,
                          GPTJConfig, MistralConfig, MixtralConfig, OPTConfig,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer, AutoModelForCausalLM,
                          AutoTokenizer)


def bloom():
    return BloomConfig(
        vocab_size=4096,
        hidden_size=32,
        n_layer=2,
        n_head=16,
    )


def gpt2():
    return GPT2Config(
        vocab_size=4096,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        # Default eos/bos are larger than tiny vocab size
        bos_token_id=0,
        eos_token_id=0,
    )


def llama():
    return LlamaConfig(
        vocab_size=4096,
        hidden_size=32,
        intermediate_size=7,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


def gptneox():
    return GPTNeoXConfig(
        vocab_size=4096,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=8,  # MLP padding not supported
        max_position_embeddings=128,
    )


def gptj():
    return GPTJConfig(
        vocab_size=4096,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        rotary_dim=16,
    )


def mistral():
    return MistralConfig(
        vocab_size=4096,
        hidden_size=32,
        intermediate_size=7,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


def mixtral():
    return MixtralConfig(
        vocab_size=4096,
        hidden_size=32,
        intermediate_size=7,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_local_experts=2,
    )


def opt():
    return OPTConfig(
        vocab_size=4096,
        hidden_size=32,
        num_hidden_layers=2,
        ffn_dim=7,
        num_attention_heads=4,
        max_position_embeddings=128,
        word_embed_proj_dim=32,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )


def build_tiny_tokenizer(model_type: str):
    models = tokenizer_from_config()
    model_name = models[model_type]
    vocab_keep_items = 4096
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer.is_fast, "This only works for fast tokenizers."
    vocab = tokenizer.get_vocab()
    if "gpt2" in model_name:
        new_vocab = {
            token: i
            for token, i in vocab.items() if i < vocab_keep_items - 1
        }
        new_vocab["<|endoftext|>"] = vocab_keep_items - 1
    else:
        new_vocab = {
            token: i
            for token, i in vocab.items() if i < vocab_keep_items
        }
    training_corpus = [new_vocab.keys()]
    new_tokenizer = tokenizer.train_new_from_iterator(
        training_corpus, vocab_size=vocab_keep_items)
    return new_tokenizer


def config_classes() -> Dict[str, Callable]:
    """
    A map from the config class to a function which builds a tiny config.

    Arguments:
        old: Whether to include old-style classes (GPT-J, GPTNeoX). This
            argument is useful because many tests are only applicable to
            classes which have been updated to use the primary TNx base
            class.

    Returns:
        configs: A mapping from config class to config instance builder
    """

    # Modernized models
    configs = {
        "bloom": bloom,
        "llama": llama,
        "gpt2": gpt2,
        "mistral": mistral,
        "mixtral": mixtral,
        "opt": opt,
        "gptneox": gptneox,
        "gptj": gptj,
    }

    return configs


def tokenizer_from_config() -> Dict[str, str]:
    """
    A map from the config class to a function which builds a tiny config.

    Arguments:
        old: Whether to include old-style classes (GPT-J, GPTNeoX). This
            argument is useful because many tests are only applicable to
            classes which have been updated to use the primary TNx base
            class.

    Returns:
        configs: A mapping from config class to config instance builder
    """

    # Modernized models
    configs = {
        "bloom": "bigscience/bloom-1b7",
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpt2": "openai-community/gpt2",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "opt": "facebook/opt-125m",
        "gptneox": "EleutherAI/gpt-neox-20b",
        "gptj": "EleutherAI/gpt-j-6b",
    }

    return configs


def config(model_type: str) -> PretrainedConfig:
    """
    Get a pre-defined tiny model config from a given config class.

    These configurations are intended to be used either in TP1 or TP2.

    Arguments:
        config_class: The class to build a config from.

    Returns:
        config: An instance of the configuration class with tiny parameters.
    """
    classes = config_classes()
    return classes[model_type]()


def build(cfg: PretrainedConfig) -> PreTrainedModel:
    """
    Build a random model from a config instance.

    Arguments:
        cfg: A huggingface pretrained model config class instance.

    Returns:
        model: An instance of the randomly initialized model.
    """
    instance = AutoModelForCausalLM.from_config(cfg)
    instance.post_init()
    instance.eval()
    return instance


@functools.lru_cache()
def artifacts(model_type: str) -> str:
    """
    Return the complete set of artifacts for a model class.

    Args:
        config_class: The class to build a artifacts for.

    Returns:
        config: The instance of the configuration class with tiny parameters.
        model: The instance of the randomly initialized tiny model.
        checkpoint: The directory where the model was saved.
    """
    cfg = config(model_type)
    tokenizer = build_tiny_tokenizer(model_type)
    model = build(cfg)
    checkpoint = tempfile.TemporaryDirectory(
        prefix=f'tiny-models-{cfg.model_type}-').name
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint
