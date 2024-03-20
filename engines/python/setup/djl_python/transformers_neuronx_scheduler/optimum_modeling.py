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
from typing import Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils import ModelOutput
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import PretrainedConfig
from optimum.neuron.generation import TokenSelector
from optimum.neuron.utils.version_utils import check_compiler_compatibility, get_neuronxcc_version
from optimum.modeling_base import OptimizedModel


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
        self.cur_len = 0

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

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.Tensor,
                                      attention_mask: Optional[
                                          torch.Tensor] = None,
                                      **kwargs) -> Dict[str, torch.Tensor]:
        # convert attention_mask to start_ids
        start_ids = None
        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)

        if self.cur_len > 0:
            # Only pass the last tokens of each sample
            input_ids = input_ids[:, -1:]
            # Specify the single index at which the new keys and values need to be stored
            cache_ids = torch.as_tensor([self.cur_len], dtype=torch.int32)
        else:
            # cache_ids will be set directly by the parallel context encoding code
            cache_ids = None

        # Increment the current cache index
        self.cur_len += input_ids.shape[-1]
        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs

    def can_generate(self) -> bool:
        """Returns True to validate the check made in `GenerationMixin.generate()`."""
        return True

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
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

        Returns:
            `torch.Tensor`: A  `torch.FloatTensor`.
        """
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(
            self.generation_config if generation_config is
            None else generation_config)
        model_kwargs = generation_config.update(
            **kwargs)  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = TokenSelector.create(input_ids, generation_config, self,
                                        self.max_length)

        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size:
            padding_shape = [self.batch_size - batch_size, sequence_length]
            padding = torch.full(padding_shape,
                                 fill_value=self.config.eos_token_id,
                                 dtype=torch.int64)
            padded_input_ids = torch.cat([input_ids, padding])
            if attention_mask is not None:
                padding = torch.zeros(padding_shape, dtype=torch.int64)
                padded_attention_mask = torch.cat([attention_mask, padding])
        # Drop the current generation context and clear the Key/Value cache
        self.reset_generation()

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            attention_mask=padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]

    def generate_tokens(
        self,
        input_ids: torch.LongTensor,
        selector: TokenSelector,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generate tokens using sampling or greedy search.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            selector (`TokenSelector`):
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

        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, attention_mask, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens = selector.select(input_ids, next_token_logits)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + selector.pad_token_id * (
                1 - unfinished_sequences)

            # update inputs for the next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                                           dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences * next_tokens.ne(
                selector.eos_token_id)

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

            # stop if we exceed the maximum length
            if selector.stopping_criteria(input_ids, None):
                break

        return input_ids
