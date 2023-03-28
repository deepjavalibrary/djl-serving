import torch
import logging
import json
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
            raise ValueError(f"{execution_engine} engine is not supported for streaming")


    @staticmethod
    @torch.inference_mode()
    def _hf_model_stream_generator(model, tokenizer, inputs, **kwargs):
        StreamingUtils._validate_inputs(inputs)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(
                                        StreamingUtils._get_current_device())
        input_ids = tokenized_inputs["input_ids"]
        input_length = input_ids.shape[1]
        all_input_ids = tokenized_inputs["input_ids"]
        max_new_tokens = kwargs.get("max_new_tokens", StreamingUtils.DEFAULT_MAX_NEW_TOKENS)
        attention_mask = input_ids.new_zeros(len(inputs), input_length + max_new_tokens)
        attention_mask[:, :input_length] = tokenized_inputs["attention_mask"]
        past_key_values = None
        decoding_method = StreamingUtils._get_decoding_method(**kwargs)
        stop_generation = False
        curr_length = input_length
        new_tokens_count = 0
        while True:
            if stop_generation:
                return
            
            attention_mask_curr = attention_mask[:, :curr_length]
            outputs = model.forward(input_ids=input_ids,
                                attention_mask=attention_mask_curr,
                                past_key_values=past_key_values,
                                use_cache=True
                                )
            next_token_ids = []
            for logits in outputs.logits:
                next_token_id = decoding_method(logits, all_input_ids, **kwargs)
                next_token_ids.append(next_token_id.view(1,1))
            token_ids = torch.cat(next_token_ids)
            past_key_values = outputs.past_key_values
            token_text = tokenizer.batch_decode(token_ids)
            input_ids = token_ids.view(len(inputs), 1)
            all_input_ids = torch.cat([all_input_ids, token_ids], dim=1)
            attention_mask[:, curr_length] = 1
            curr_length += 1
            new_tokens_count += 1
            stop_generation = StreamingUtils._has_met_stopping_criteria(token_ids, new_tokens_count,
                                        max_new_tokens, model.config.eos_token_id)
            yield token_text


    @staticmethod
    @torch.inference_mode()
    def _transformers_neuronx_stream_generator(model, tokenizer, inputs, **kwargs):
        sequence_length = kwargs.get("seq_len", 50)
        top_k =kwargs.get("top_k", 50)
        input_ids = torch.as_tensor([tokenizer.encode(text) for text in inputs])
        model.reset()
        eos_token_id = model.config.eos_token_id
        # populate key/value caches according to the prompt text
        _, start = input_ids.shape
        position_ids = torch.arange(start, dtype=torch.int32)
        next_token_scores = model(input_ids, position_ids)

        tokens = [input_ids]
        _, start = input_ids.shape
        for cur_len in range(start, sequence_length):
            # don't sample EOS
            next_token_scores[:, eos_token_id] = -float('inf')

            # Remove all tokens with a probability less than the last token of the top-k
            topk_values, topk_indices = torch.topk(next_token_scores, top_k)
            probs = torch.nn.functional.softmax(topk_values, dim=-1)
            inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
            inputs = torch.gather(topk_indices, 1, inputs_in_topk)
            tokens.append(inputs)
            token_text = tokenizer.decode(inputs[0][0])
            position_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = model(inputs, position_ids)
            yield token_text


    @staticmethod
    def _has_met_stopping_criteria(token, current_token_count, max_new_tokens, eos_token_id):
        if token == eos_token_id or current_token_count >= max_new_tokens:
            return True
        return False


    def _validate_inputs(inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 1, "[ERROR] batching is not yet supported"
        else:
            assert False, "inputs to stream generator must be a list of strings"


    def _greedy_decoding(logits, input_ids, **kwargs):
        processors = LogitsProcessorList()
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=kwargs["repetition_penalty"]))

        logits[-1:, :] = processors(input_ids, logits[-1:, :])
        return logits[-1].argmax()


    def _sampling_decoding(logits, input_ids, **kwargs):
        processors = LogitsProcessorList()
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=kwargs["repetition_penalty"]))
        if "temperature" in kwargs and kwargs["temperature"] != 1.0:
            processors.append(TemperatureLogitsWarper(float(kwargs["temperature"])))
        if "top_p" in kwargs and kwargs["top_p"] < 1.0:
            processors.append(TopPLogitsWarper(kwargs["top_p"]))
        if "top_k" in kwargs and kwargs["top_k"] != 0:
            processors.append(TopKLogitsWarper(kwargs["top_k"]))
        if "typical_p" in kwargs and kwargs["typical_p"] < 1.0:
            processors.append(TypicalLogitsWarper(mass=kwargs["typical_p"]))

        logits[-1:, :] = processors(input_ids, logits[-1:, :])
        generator = torch.Generator(StreamingUtils.DEVICE)
        if "manual_seed" in kwargs:
            generator.manual_seed(kwargs["manual_seed"])
        probs = torch.nn.functional.softmax(logits[-1])
        return torch.multinomial(probs, num_samples=1, generator=generator)
        

    def _get_decoding_method(**kwargs):
        if "beam_size" in kwargs:
            logging.warning("beam search is not supported yet, using greedy search instead.")
            return StreamingUtils._greedy_decoding
        elif any(param in kwargs for param in ["temperature", "top_p", "top_k", "typical_p"]):
            return StreamingUtils._sampling_decoding
        else:
            return StreamingUtils._greedy_decoding

    def _get_current_device():
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")
        