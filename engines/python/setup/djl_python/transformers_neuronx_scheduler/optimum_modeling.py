#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
# The below code is heavily inspired from Optimum Neuron under the following link:
# https://github.com/huggingface/optimum-neuron/blob/v0.0.16-release/optimum/neuron/modeling.py

import copy
import torch
from typing import Dict, Optional, Union, List
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils import ModelOutput
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import PretrainedConfig
from transformers_neuronx import bucket
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from optimum.neuron.utils.version_utils import check_compiler_compatibility, get_neuronxcc_version
from djl_python.transformers_neuronx_scheduler.optimum_token_selector import OptimumTokenSelector
from optimum.modeling_base import OptimizedModel
from transformers.generation import StoppingCriteriaList


class OptimumModelForCausalLM(OptimizedModel, GenerationMixin):
    """
        Overwrite of the NeuronModelForCausalLM to match legacy interface while bringing in new model generation methods.
        Base class to convert and run pre-trained transformers decoder models on Neuron devices.

        It implements the methods to convert a pre-trained transformers decoder model into a Neuron transformer model by:
        - transferring the checkpoint weights of the original into an optimized neuron graph,
        - compiling the resulting graph using the Neuron compiler.

        Common attributes:
            - model (`torch.nn.Module`) -- The decoder model with a graph optimized for neuron devices.
            - config ([`~transformers.PretrainedConfig`]) -- The configuration of the original model.
            - generation_config ([`~transformers.GenerationConfig`]) -- The generation configuration used by default when calling `generate()`.
        """
    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    CHECKPOINT_DIR = "checkpoint"
    COMPILED_DIR = "compiled"

    def __init__(self,
                 model: torch.nn.Module,
                 config: "PretrainedConfig",
                 model_path: Union[str, "Path", "TemporaryDirectory"],
                 generation_config: Optional[GenerationConfig] = None,
                 **kwargs) -> None:
        super().__init__(model, config)
        self.model_path = model_path
        self.cur_len = 0
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config
        neuron_config = getattr(config, "neuron", None)
        check_compiler_compatibility(neuron_config["compiler_type"],
                                     neuron_config["compiler_version"])
        self.batch_size = self.config.neuron["batch_size"]
        self.max_length = self.config.neuron["sequence_length"]
        # The generate method from GenerationMixin expects the device attribute to be set
        self.device = torch.device("cpu")

    def reset_generation(self) -> None:
        pass

    def _save_pretrained(self, save_directory):
        """
        Saves a model weights into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_ids: torch.Tensor,
        start_ids: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # Evaluate the output logits, storing the current key and values at the indices specified by cache_ids
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        # Since we are using a static cache, we don't need to return past keys and values
        if return_dict:
            return ModelOutput([("logits", out_logits)])
        return (out_logits, )

    def speculative_forward(self, *args, **kwargs):
        if hasattr(self.model, "speculative_forward"):
            # Workaround until model.speculative_forward accuracy is fixed for llama
            return self.model.speculative_forward(*args, **kwargs)
        else:
            raise NotImplementedError(
                "Model does not support speculative forward")

    def get_start_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[torch.Tensor] = None,
    ):
        # The start_ids parameter has different meanings:
        # - for static batching it corresponds to the start of the padded sequence.
        start_ids = None
        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)
        return start_ids

    def get_cache_ids(self, attention_mask: torch.tensor, prefill: bool):
        cache_n, cache_len = attention_mask.shape
        # Static batching
        return None if prefill else torch.tensor([cache_len - 1],
                                                 dtype=torch.int32)

    def prepare_inputs_for_prefill(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            seq_ids: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids,
                                       attention_mask,
                                       seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=True)
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def prepare_inputs_for_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids,
                                       attention_mask,
                                       seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=False)
        # Only pass the last tokens of each sample
        input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def can_generate(self) -> bool:
        """Returns True to validate the check made in `GenerationMixin.generate()`."""
        return True

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

        This method uses the same logits processors/warpers and stopping criterias as the transformers library
        `generate()` method but restricts the generation to greedy search and sampling.

        It does not support transformers `generate()` advanced options.

        Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        for details on generation configuration.

        Parameters:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~transformers.generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.

        Returns:
            `torch.Tensor`: A  `torch.FloatTensor`.
        """
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(
            self.generation_config if generation_config is
            None else generation_config)
        # Extract tokenizer if any (used only for stop strings)
        tokenizer = kwargs.pop("tokenizer", None)
        model_kwargs = generation_config.update(
            **kwargs)  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = OptimumTokenSelector.create(
            input_ids,
            generation_config,
            self,
            self.max_length,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = torch.ones_like(
            input_ids) if attention_mask is None else attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size:
            logger.warning(
                "Inputs will be padded to match the model static batch size. This will increase latency."
            )
            padding_shape = [self.batch_size - batch_size, sequence_length]
            pad_token_id = generation_config.pad_token_id
            if pad_token_id is None:
                if isinstance(self.config.eos_token_id, list):
                    pad_token_id = self.config.eos_token_id[0]
                else:
                    pad_token_id = self.config.eos_token_id
            padding = torch.full(padding_shape,
                                 fill_value=pad_token_id,
                                 dtype=torch.int64)
            padded_input_ids = torch.cat([padded_input_ids, padding])
            padding = torch.zeros(padding_shape, dtype=torch.int64)
            padded_attention_mask = torch.cat([padded_attention_mask, padding])

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]

    def generate_tokens(
        self,
        input_ids: torch.LongTensor,
        selector: OptimumTokenSelector,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generate tokens using sampling or greedy search.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            selector (`OptimumTokenSelector`):
                The object implementing the generation logic based on transformers processors and stopping criterias.
            batch_size (`int`):
                The actual input batch size. Used to avoid generating tokens for padded inputs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens.

        """
        # keep track of which sequences are already finished
        unfinished_sequences = torch.zeros(input_ids.shape[0],
                                           dtype=torch.long,
                                           device=input_ids.device)
        unfinished_sequences[:batch_size] = 1

        # Prefill and obtain the first token
        model_inputs = self.prepare_inputs_for_prefill(input_ids,
                                                       attention_mask)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
        )

        # auto-regressive generation
        while True:
            next_token_logits = outputs.logits[:, -1, :]

            next_tokens = selector.select(input_ids, next_token_logits)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + selector.pad_token_id * (
                1 - unfinished_sequences)

            # update inputs for the next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))
            ],
                                       dim=-1)

            unfinished_sequences = unfinished_sequences & ~selector.stopping_criteria(
                input_ids, None)

            if unfinished_sequences.max() == 0:
                break

            # forward pass to get next token
            model_inputs = self.prepare_inputs_for_decode(
                input_ids, attention_mask)
            outputs = self.model(
                **model_inputs,
                return_dict=True,
            )

        return input_ids
