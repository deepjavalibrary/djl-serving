import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('model',
                    help='s3 bucket url')

model_list = {
    "bloom-7b1" : { "option.s3url" :"s3://djl-llm/bloom-7b1/", "option.tensor_parallel_degree" : 4 },
    "opt-30b" : { "option.s3url" :"s3://djl-llm/opt-30b/", "option.tensor_parallel_degree" : 4 }
}

args = parser.parse_args()

if args.model not in model_list:
    raise ValueError(f"{args.model} is not one of the supporting models {list(model_list.keys())}")

options = model_list[args.model]
options["engine"] = "DeepSpeed"

os.makedirs("models/test", exist_ok=True)
with open("models/test/serving.properties", "w") as f:
    for key, value in options.items():
        f.write(f"{key}={value}\n")
shutil.copyfile("deepspeed/deepspeed-model.py", "models/test/model.py")
