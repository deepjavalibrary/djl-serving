import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('model',
                    help='s3 bucket url')

model_list = {
    "gpt-j-6b" : { "option.s3url" :"s3://djl-llm/gpt-j-6b/", "option.tensor_parallel_degree" : 4 },
    "bloom-7b1" : { "option.s3url" :"s3://djl-llm/bloom-7b1/", "option.tensor_parallel_degree" : 4, "option.dtype": "float16" },
    "opt-30b" : { "option.s3url" :"s3://djl-llm/opt-30b/", "option.tensor_parallel_degree" : 4 }
}

args = parser.parse_args()

if args.model not in model_list:
    raise ValueError(f"{args.model} is not one of the supporting models {list(model_list.keys())}")

options = model_list[args.model]
options["engine"] = "DeepSpeed"

model_path = "models/test"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

os.makedirs(model_path, exist_ok=True)
with open(os.path.join(model_path, "serving.properties"), "w") as f:
    for key, value in options.items():
        f.write(f"{key}={value}\n")
shutil.copyfile("deepspeed/deepspeed-model.py", "models/test/model.py")
