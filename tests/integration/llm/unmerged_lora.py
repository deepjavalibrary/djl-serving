from djl_python import Input, Output
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import json
import logging

BASE_MODEL_ID = "huggyllama/llama-7b"
LORA_ADAPTER_1_ID = "tloen/alpaca-lora-7b"
LORA_ADAPTER_2_ID = "22h/cabrita-lora-v0-1"
LORA_ADAPTER_1_NAME = "english-alpaca"
LORA_ADAPTER_2_NAME = "protugese-alpaca"

model = None
tokenizer = None


def construct_error_output(output, err_msg):
    error = {"code": 500, "error": err_msg}
    error = json.dumps(error)
    output.add(error, key="data")
    return output


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the 
    request.### Instruction: {instruction} ### Response:"""


def load_model():
    global model, tokenizer
    # load base model
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16).to("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_ID)
    if not tokenizer.pad_token:
        tokenizer.pad_token = '[PAD]'

    # load lora adapter 1
    model = PeftModel.from_pretrained(model,
                                      LORA_ADAPTER_1_ID,
                                      adapter_name=LORA_ADAPTER_1_NAME)
    # load lora adapter 2
    model.load_adapter(LORA_ADAPTER_2_ID, adapter_name=LORA_ADAPTER_2_NAME)


def inference():
    global model, tokenizer
    output = Output()
    if len(model.peft_config.keys()) != 2:
        return construct_error_output(
            output, "Incorrect number of adapters registered")

    input1 = {
        "inputs": "Tell me about Alpacas",
        "adapter_name": LORA_ADAPTER_1_NAME
    }
    input2 = {
        "inputs":
        "Invente uma desculpa criativa pra dizer que não preciso ir à festa.",
        "adapter_name": LORA_ADAPTER_2_NAME
    }

    generation_config = GenerationConfig(num_beams=1, do_sample=False)

    prompts = [
        generate_prompt(input1["inputs"]),
        generate_prompt(input2["inputs"]),
    ]

    adapters = [
        input1["adapter_name"],
        input2["adapter_name"],
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(torch.cuda.current_device())
    attention_mask = inputs["attention_mask"].to(torch.cuda.current_device())
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        adapters=adapters,
        generation_config=generation_config,
        return_dict_in_generate=False,
        max_new_tokens=64,
    )
    outputs_unmerged_lora = tokenizer.batch_decode(outputs,
                                                   skip_special_tokens=True)
    if len(outputs_unmerged_lora) != 2:
        return construct_error_output(output, "Incorrect number of outputs")

    logging.info(f"outputs from unmerged lora: {outputs_unmerged_lora}")

    model.delete_adapter(LORA_ADAPTER_2_NAME)
    if len(model.peft_config.keys()) != 1:
        return construct_error_output(
            output, "Incorrect number of adapters registered after delete op")

    # merge lora adapter 1 into base model
    model.set_adapter(LORA_ADAPTER_1_NAME)
    model.merge_and_unload()
    outputs_lora_1 = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=False,
        max_new_tokens=64,
    )

    outputs_merged_lora = tokenizer.batch_decode(outputs_lora_1,
                                                 skip_special_tokens=True)

    prediction = [{
        'unmerged_lora_result': outputs_unmerged_lora[0]
    }, {
        'merged_lora_result': outputs_merged_lora[0]
    }]
    output.add_as_json(prediction, key="data")
    return output


def handle(input: Input):
    if not model:
        load_model()

    if input.is_empty():
        return None

    return inference()
