import torch
import tempfile
import os

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers_neuronx import dtypes
from transformers_neuronx.gpt2.model import GPT2ForSampling
from djl_python import Input, Output

model = None


def load_model(properties):
    batch_size = int(properties.get("batch_size", 4))
    tp_degree = int(properties.get("tensor_parallel_degree", 2))
    amp = properties.get("dtype", "f32")
    n_positions = int(properties.get("n_positions", 128))
    unroll = properties.get("unroll", None)
    model_id = "gpt2"
    load_path = os.path.join(tempfile.gettempdir(), model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 low_cpu_mem_usage=True)
    dtype = dtypes.to_torch_dtype(amp)
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)
    model.save_pretrained(load_path, max_shard_size="100GB")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = GPT2ForSampling.from_pretrained(load_path,
                                            batch_size=batch_size,
                                            amp=amp,
                                            tp_degree=tp_degree,
                                            n_positions=n_positions,
                                            unroll=unroll)
    model.to_neuron()
    return model, tokenizer, batch_size


def infer(seq_length, prompt):
    with torch.inference_mode():
        encoded_text = tokenizer.encode(prompt)
        input_ids = torch.as_tensor([encoded_text])
        input_ids = torch.cat([input_ids for _ in range(batch_size)], dim=0)
        generated_sequence = model.sample(input_ids,
                                          sequence_length=seq_length)
        outputs = [tokenizer.decode(gen_seq) for gen_seq in generated_sequence]
    return outputs


def handle(inputs: Input):
    global model, tokenizer, batch_size
    if not model:
        model, tokenizer, batch_size = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    seq_length = data["seq_length"]
    prompt = data["text"]
    outputs = infer(seq_length, prompt)
    result = {"outputs": outputs}
    return Output().add_as_json(result)
