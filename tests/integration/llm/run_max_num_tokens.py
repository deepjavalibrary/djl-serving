import argparse
import json
import os
import urllib.request
from tensorrt_llm_toolkit.utils.utils import max_token_finder
from pathlib import Path

if __name__ == '__main__':
    os.mkdir("max_num_token_results")

    parser = argparse.ArgumentParser(description='Parse inputs to script')
    parser.add_argument('i_model_tp_json',
                        help='Input - JSON containing model and tp pairs')
    args = parser.parse_args()

    data = urllib.request.urlopen(args.i_model_tp_json)
    json_str = [line.decode("utf-8").strip() for line in data][0]
    model_tp_dict = json.loads(json_str)

    log_id = 0
    for model_id, tp_list in model_tp_dict.items():
        model_name = model_id
        print(f"Starting runs for model {model_name}:")
        if model_id[:2] == 's3':
            # download model
            s3url = model_id
            if Path("/opt/djl/bin/s5cmd").is_file():
                if not s3url.endswith("*"):
                    if s3url.endswith("/"):
                        s3url = s3url + '*'
                    else:
                        s3url = s3url + '/*'
                os.system(
                    f"/opt/djl/bin/s5cmd --retry-count 1 sync {s3url} /tmp/model/"
                )
            else:
                os.system(f"aws s3 cp {s3url} /tmp/model/ --recursive")
            model_id = "/tmp/model/"
        if isinstance(tp_list, int):
            tp_list = [tp_list]
        for tensor_parallel_degree in tp_list:
            print(
                f"Starting run for model {model_name} with tp={tensor_parallel_degree}:"
            )
            properties = {
                "model_id": model_id,
                "tensor_parallel_degree": tensor_parallel_degree,
            }
            model, tp, max_tokens = max_token_finder(properties)
            output = f"Summary:\nmodel: {model_name}\n tp: {tp}\n max_tokens: {max_tokens}"
            print(output)
            with open("max_num_token_results/" + str(log_id) + "_log.txt",
                      "w") as log_file:
                log_file.write(output)
            log_id += 1
