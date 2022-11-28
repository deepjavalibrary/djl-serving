import requests
import argparse

parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('model',
                    help='The name of model')

endpoint="http://127.0.0.1:8080/predictions/test"

model_spec = {
    "opt-30b" : {"max_memory_per_gpu" : 10.0, "batch_size" : [1, 2, 4, 8], "seq_length" : [64, 128, 256] }
}

def send_json(data):
    headers = {'content-type': 'application/json'}
    res = requests.post(endpoint, headers=headers, json=data)
    return res.json()


def test_model(model):
    if model not in model_spec:
        raise ValueError(f"{args.model} is not one of the supporting models {list(model_spec.keys())}")
    spec = model_spec[args.model]
    for batch_size in spec["batch_size"]:
        for seq_length in spec["seq_length"]:
            req = {"batch_size" : batch_size, "text_length" : seq_length }
            res = send_json(req)
            print(res)
            assert len(res["outputs"]) == batch_size
            assert float(res["reserved_memory"]) < spec["max_memory_per_gpu"]


args = parser.parse_args()
test_model(args.model)


