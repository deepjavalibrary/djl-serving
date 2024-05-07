import argparse
from tensorrt_llm_toolkit.utils.utils import max_token_finder


model_map = {"Llama 2 7B":"TheBloke/Llama-2-7B-fp16",
             "Llama 2 13B":"TheBloke/Llama-2-13B-fp16",
             "Llama 2 70B":"TheBloke/Llama-2-70B-fp16"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse inputs to script')
    parser.add_argument('i_model_id', help='Input - model id')
    parser.add_argument('i_tp_degree', help='Input - tp degree')
    args = parser.parse_args()
    
    properties = {
        "model_id": model_map[args.i_model_id],
        "tensor_parallel_degree": args.i_tp_degree,
    }
    
    model, tp, max_tokens = max_token_finder(properties)
    print(f"Summary:\nmodel: {model}\n tp: {tp}\n max_tokens: {max_tokens}")
