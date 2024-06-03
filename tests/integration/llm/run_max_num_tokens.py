import argparse
import json
import os
import urllib.request
from tensorrt_llm_toolkit.utils.utils import max_token_finder
from pathlib import Path


def clear_artifacts():
    os.system("rm -rf /tmp/trtllm*")
    os.system("rm -rf /tmp/model")
    os.system("rm -rf /tmp/.djl.ai*")
    os.mkdir("/tmp/model")


if __name__ == '__main__':
    print("listing temp:")
    os.system("ls /tmp")
    print("making a file:")
    os.system('echo "hello_world" >> /tmp/test.txt')
    print("listing temp again:")
    os.system('ls /tmp')
    os.mkdir("max_num_token_results")

    parser = argparse.ArgumentParser(description='Parse inputs to script')
    parser.add_argument('i_model_tp_json',
                        help='Input - JSON containing model and tp pairs')
    args = parser.parse_args()

    data = urllib.request.urlopen(args.i_model_tp_json)
    json_str = "".join([line.decode("utf-8").strip() for line in data])
    model_tp_dict = json.loads(json_str)
    os.mkdir("/tmp/model/")

    for model_id, tp_list in model_tp_dict.items():
        model_name = os.path.basename(os.path.normpath(model_id))
        print(f"Starting runs for model {model_name}:")
        if model_id[:2] == 's3':
            # download model
            num_tries = 0
            while len(os.listdir("/tmp/model/")) < 3 and num_tries < 5:
                print(f"Downloading from s3 (try {num_tries}...")
                clear_artifacts()
                s3url = model_id
                if Path("/opt/djl/bin/s5cmd").is_file() and num_tries < 2:
                    print("Using s5cmd...")
                    if not s3url.endswith("*"):
                        if s3url.endswith("/"):
                            s3url = s3url + '*'
                        else:
                            s3url = s3url + '/*'
                    return_value = os.system(
                        f"/opt/djl/bin/s5cmd --retry-count 1 sync {s3url} /tmp/model/"
                    )
                    print("Return value from s5cmd: ", return_value)
                else:
                    print("Using AWS CLI...")
                    return_value = os.system(
                        f"aws s3 cp {s3url} /tmp/model/ --recursive")
                    print("Return value from aws s3 cp: ", return_value)
                model_id = "/tmp/model/"
                num_tries += 1
        if len(os.listdir("/tmp/model/")) < 3:
            # Still failed
            print(
                "Model download failed after multiple tries: directory had these contents:"
            )
            print(os.listdir("/tmp/model/"))
            clear_artifacts()
            with open(f"max_num_token_results/{model_name}_log.txt",
                      "w") as log_file:
                log_file.write("{model_name} model download failed")
            continue
        if isinstance(tp_list, int):
            tp_list = [tp_list]
        for tensor_parallel_degree in tp_list:
            print(
                f"Starting run for model {model_name} with tp={tensor_parallel_degree}:"
            )
            properties = {
                "model_id": model_id,
                "tensor_parallel_degree": tensor_parallel_degree,
                "trust_remote_code": "true"
            }
            try:
                model, tp, max_tokens = max_token_finder(properties)
            except:
                max_tokens, tp = -1, tensor_parallel_degree
            output = f"Summary:\nmodel: {model_name}\n tp: {tensor_parallel_degree}\n max_tokens: {max_tokens}"
            print(output)
            with open(
                    f"max_num_token_results/{model_name}_{tensor_parallel_degree}_log.txt",
                    "w") as log_file:
                log_file.write(output)
        clear_artifacts()
