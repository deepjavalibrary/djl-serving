from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig

torch.manual_seed(1234)


def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    checkpoint = None
    if "checkpoint" in properties:
        checkpoint = os.path.join(model_location, properties['checkpoint'])

    if checkpoint:
        config_file = os.path.join(model_location, "config.json")
        config = AutoConfig.from_pretrained(config_file)
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_location,
                                                 low_cpu_mem_usage=True)
    if "dtype" in properties:
        if properties["dtype"] == "float16":
            model.to(torch.float16)
        if properties["dtype"] == "bfloat16":
            model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    logging.info(f"Starting DeepSpeed init with TP={tensor_parallel}")
    model = deepspeed.init_inference(model,
                                     tensor_parallel={"tp_size": tensor_parallel},
                                     dtype=model.dtype,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True,
                                     save_mp_checkpoint_path=properties.get("save_mp_checkpoint_path"))
    return model.module, tokenizer


def batch_generation(batch_size):
    input_sentences = [
        "DeepSpeed is a machine learning framework",
        "He is working on",
        "He has a",
        "He got all",
        "Everyone is happy and I can",
        "The new movie that got Oscar this year",
        "In the far far distance from our galaxy,",
        "Peace is the only way",
    ]
    if batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    return input_sentences[:batch_size]


model = None
tokenizer = None
generator = None


def separate_inference(model, tokenizer, batch_size, length):
    generate_kwargs = dict(max_new_tokens=length, do_sample=True)
    input_tokens = tokenizer.batch_encode_plus(batch_generation(batch_size),
                                               return_tensors="pt",
                                               padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    outputs = model.generate(**input_tokens, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def pipeline_inference(model, tokenizer, batch_size, length):
    global generator
    if not generator:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        generator = pipeline(task='text-generation',
                             model=model,
                             tokenizer=tokenizer,
                             device=local_rank)
    outputs = generator(batch_generation(batch_size), max_length=length)
    return [item[0]['generated_text'] for item in outputs]


def partition(inputs: Input):
    load_model(inputs.get_properties())

def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    data = inputs.get_as_json()
    batch_size = data["batch_size"]
    tokens_to_gen = data["text_length"]
    if data["use_pipeline"]:
        outputs = pipeline_inference(model, tokenizer, batch_size,
                                     tokens_to_gen)
    else:
        outputs = separate_inference(model, tokenizer, batch_size,
                                     tokens_to_gen)
    result = {"outputs": outputs}
    return Output().add_as_json(result)
