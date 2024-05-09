import argparse
import json
from tensorrt_llm_toolkit.utils.utils import max_token_finder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse inputs to script')
    parser.add_argument('i_model_tp_json', help='Input - JSON containing model and tp pairs')
    args = parser.parse_args()

    with open(args.i_model_tp_json, "r") as file:
        model_tp_dict = json.load(file)

    for model_id, tp_list in model_tp_dict.items():
        if isinstance(tp_list, int):
            tp_list = [tp_list]
        for tensor_parallel_degree in tp_list:
            properties = {
                "model_id": model_id,
                "tensor_parallel_degree": tensor_parallel_degree,
            }
            model, tp, max_tokens = max_token_finder(properties)
            print(f"Summary:\nmodel: {model}\n tp: {tp}\n max_tokens: {max_tokens}")
            # save files and then upload todo
