import torch
import logging
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)


class StreamingUtils:

    DEFAULT_MAX_NEW_TOKENS = 50
    SUPPORTED_MODEL_ARCH_SUFFIXES_CAUSAL_LM = ("CausalLM", "GPT2LMHeadModel")
    SUPPORTED_MODEL_ARCH_SUFFIXES_SEQ_2_SEQ_LM = ("T5ForConditionalGeneration",)
    SUPPORTED_MODEL_ARCH_SUFFIXES = SUPPORTED_MODEL_ARCH_SUFFIXES_CAUSAL_LM + SUPPORTED_MODEL_ARCH_SUFFIXES_SEQ_2_SEQ_LM

    @staticmethod
    def get_stream_generator(execution_engine: str):
        ## execution_engine passed to this function is not the same engine specified in serving.properties
        ## in djl-serving. For e.g Accelerate and neuronx use Python as the engine serving.properties
        ## The engine here refers to backend model parallel framework.
        if execution_engine in {"DeepSpeed", "Accelerate"}:
            return StreamingUtils._hf_model_stream_generator
        elif execution_engine == "transformers-neuronx":
            return StreamingUtils._transformers_neuronx_stream_generator
        else:
            raise ValueError(
                f"{execution_engine} engine is not supported for streaming")

    @staticmethod
    @torch.inference_mode()
    def _hf_model_stream_generator(model, tokenizer, inputs, **kwargs):
        StreamingUtils._validate_inputs(model, inputs)
        generic_model_class = StreamingUtils._get_generic_model_class(model)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        if generic_model_class == "Seq2SeqLM":
            tokenizer.bos_token_id = model.config.decoder_start_token_id

        max_new_tokens = kwargs.get("max_new_tokens",
                                    StreamingUtils.DEFAULT_MAX_NEW_TOKENS)
        tokenized_inputs = tokenizer(inputs, return_tensors="pt",
                                     padding=True).to(
                                         StreamingUtils._get_current_device())
        input_ids = tokenized_inputs["input_ids"]
        past_key_values = None
        decoding_method = StreamingUtils._get_decoding_method(**kwargs)
        new_tokens_count = 0
        unfinished_sequences = torch.ones((len(inputs), 1),
                                          dtype=torch.long,
                                          device=input_ids.device)
        stop_generation = False

        if generic_model_class == "CausalLM":
            input_length = input_ids.shape[1]
            all_decoder_input_ids = tokenized_inputs["input_ids"]
            is_pad_token_equal_to_eos_token = tokenizer.pad_token == tokenizer.eos_token
            attention_mask = input_ids.new_zeros(len(inputs),
                                                 input_length + max_new_tokens)
            attention_mask[:, :
                           input_length] = 1 if is_pad_token_equal_to_eos_token else tokenized_inputs[
                               "attention_mask"]
            curr_length = input_length

        if generic_model_class == "Seq2SeqLM":
            attention_mask = tokenized_inputs["attention_mask"]
            decoder_attention_mask = None
            encoder_last_hidden_state = None
            decoder_input_ids = torch.tensor(tokenizer.bos_token_id,
                                    device=StreamingUtils._get_current_device()).repeat(len(inputs)).view(-1, 1)
            all_decoder_input_ids = decoder_input_ids

        while True:

            if generic_model_class == "CausalLM":
                attention_mask_curr = attention_mask[:, :curr_length]
                outputs = model.forward(input_ids=input_ids,
                                        attention_mask=attention_mask_curr,
                                        past_key_values=past_key_values,
                                        use_cache=True)

            if generic_model_class == "Seq2SeqLM":
                outputs = model.forward(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        encoder_outputs=encoder_last_hidden_state,
                                        past_key_values=past_key_values,
                                        use_cache=True)

            next_token_ids = []
            ## TODO: batch decoding
            for i, logits in enumerate(outputs.logits):
                next_token_id = decoding_method(
                    logits, all_decoder_input_ids[i, :].view(1, -1), **kwargs)
                next_token_ids.append(next_token_id.view(1, 1))
            token_ids = torch.cat(next_token_ids)

            all_decoder_input_ids = torch.cat([all_decoder_input_ids, token_ids], dim=1)
            past_key_values = outputs.past_key_values
            new_tokens_count += 1

            not_eos_token_ids = (token_ids != tokenizer.eos_token_id).view(
                    len(inputs), 1)
            unfinished_sequences = unfinished_sequences.mul(not_eos_token_ids)

            if generic_model_class == "CausalLM":
                input_ids = token_ids.view(len(inputs), 1)
                input_ids = input_ids * unfinished_sequences + tokenizer.pad_token_id * unfinished_sequences.logical_not()
                token_text = tokenizer.batch_decode(input_ids)
                attention_mask[:, curr_length] = 1
                curr_length += 1

            if generic_model_class == "Seq2SeqLM":
                input_ids = None
                decoder_input_ids = token_ids.view(len(inputs), 1)
                decoder_input_ids = decoder_input_ids * unfinished_sequences + tokenizer.pad_token_id * unfinished_sequences.logical_not()
                encoder_last_hidden_state = [outputs.encoder_last_hidden_state]
                token_text = tokenizer.batch_decode(decoder_input_ids)

            stop_generation = StreamingUtils._has_met_stopping_criteria(
                unfinished_sequences, new_tokens_count, max_new_tokens)

            if stop_generation:
                return
            yield token_text


    @staticmethod
    @torch.inference_mode()
    def _transformers_neuronx_stream_generator(model, tokenizer, inputs,
                                               **kwargs):
        sequence_length = kwargs.get("seq_length",
                                     StreamingUtils.DEFAULT_MAX_NEW_TOKENS)
        top_k = kwargs.get("top_k", 50)
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
        input_ids = tokenized_inputs["input_ids"]
        model.reset()
        eos_token_id = model.config.eos_token_id
        # populate key/value caches according to the prompt text
        _, start = input_ids.shape
        position_ids = torch.arange(start, dtype=torch.int32)
        next_token_scores = model(input_ids, position_ids)

        tokens = [input_ids]
        for cur_len in range(start, sequence_length):
            # don't sample EOS
            next_token_scores[:, eos_token_id] = -float('inf')

            # Remove all tokens with a probability less than the last token of the top-k
            topk_values, topk_indices = torch.topk(next_token_scores, top_k)
            probs = torch.nn.functional.softmax(topk_values, dim=-1)
            inputs_in_topk = torch.multinomial(probs,
                                               num_samples=1,
                                               replacement=True)
            inputs = torch.gather(topk_indices, 1, inputs_in_topk)
            tokens.append(inputs)
            token_text = tokenizer.batch_decode(inputs)
            position_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = model(inputs, position_ids)
            yield token_text

    @staticmethod
    def _has_met_stopping_criteria(not_eos_token_ids, current_token_count,
                                   max_new_tokens):
        if not_eos_token_ids.sum(
        ) == 0 or current_token_count >= max_new_tokens:
            return True
        return False

    @staticmethod
    def _validate_inputs(model, inputs):
        if not model.config.architectures:
            ## do best effort validation as there is no simple way to cover all the cases
            logging.warning(f"Model config does not contain architectures field. Supported architectures: *{StreamingUtils.SUPPORTED_MODEL_ARCH_SUFFIXES}")
            model_arch_supported = True
        else:
            model_arch_list = model.config.architectures
            model_arch_supported = any(
                model_arch.endswith(StreamingUtils.SUPPORTED_MODEL_ARCH_SUFFIXES)
                for model_arch in model_arch_list)
        if not model_arch_supported:
            assert False, f"model archs: {model_arch_list} is not in supported list: *{StreamingUtils.SUPPORTED_MODEL_ARCH_SUFFIXES}"
        if isinstance(inputs, list):
            assert len(inputs) >= 1, "[ERROR] empty input list"
        else:
            assert False, "inputs to stream generator must be a list of strings"

    @staticmethod
    def _get_generic_model_class(model):
        if not model.config.architectures:
            ## do best effort validation as there is no simple way to cover all the cases
            logging.warning(f"Model config does not contain architectures field. Assuming it is CausalLM type")
            return "CausalLM"
        else:
            model_arch_list = model.config.architectures
            if any(
                model_arch.endswith(StreamingUtils.SUPPORTED_MODEL_ARCH_SUFFIXES_SEQ_2_SEQ_LM)
                    for model_arch in model_arch_list):
                return  "Seq2SeqLM"
            else:
                return "CausalLM"

    @staticmethod
    def _greedy_decoding(logits, input_ids, **kwargs):
        processors = LogitsProcessorList()
        if "repetition_penalty" in kwargs and kwargs[
                "repetition_penalty"] != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(
                    penalty=kwargs["repetition_penalty"]))
        logits[-1:, :] = processors(input_ids, logits[-1:, :])
        return logits[-1].argmax()

    @staticmethod
    def _sampling_decoding(logits, input_ids, **kwargs):
        processors = LogitsProcessorList()
        if "repetition_penalty" in kwargs and kwargs[
                "repetition_penalty"] != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(
                    penalty=kwargs["repetition_penalty"]))
        if "temperature" in kwargs and kwargs["temperature"] != 1.0:
            processors.append(
                TemperatureLogitsWarper(float(kwargs["temperature"])))
        if "top_p" in kwargs and kwargs["top_p"] < 1.0:
            processors.append(TopPLogitsWarper(kwargs["top_p"]))
        if "top_k" in kwargs and kwargs["top_k"] != 0:
            processors.append(TopKLogitsWarper(kwargs["top_k"]))
        if "typical_p" in kwargs and kwargs["typical_p"] < 1.0:
            processors.append(TypicalLogitsWarper(mass=kwargs["typical_p"]))

        logits[-1:, :] = processors(input_ids, logits[-1:, :])
        generator = torch.Generator(StreamingUtils._get_current_device())
        probs = torch.nn.functional.softmax(logits[-1])
        if "manual_seed" in kwargs:
            generator.manual_seed(kwargs["manual_seed"])
            torch.multinomial(probs, num_samples=1, generator=generator)

        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _get_decoding_method(**kwargs):
        if "beam_size" in kwargs:
            logging.warning(
                "beam search is not supported yet, using greedy search instead."
            )
            return StreamingUtils._greedy_decoding
        elif any(param in kwargs
                 for param in ["temperature", "top_p", "top_k", "typical_p"]):
            return StreamingUtils._sampling_decoding
        else:
            return StreamingUtils._greedy_decoding

    @staticmethod
    def _get_current_device():
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")
