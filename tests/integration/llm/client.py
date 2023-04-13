import requests
import argparse
import subprocess as sp
import logging
import math
import json
from io import BytesIO

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Build the LLM configs')
parser.add_argument('handler', help='the handler used in the model')
parser.add_argument('model', help='The name of model')

endpoint = "http://127.0.0.1:8080/predictions/test"

ds_raw_model_spec = {
    "gpt-j-6b": {
        "max_memory_per_gpu": [6.0, 6.0, 6.0, 6.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": True
    },
    "bloom-7b1": {
        "max_memory_per_gpu": [7.0, 7.0, 8.0, 9.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": False
    },
    "opt-30b": {
        "max_memory_per_gpu": [16.0, 16.0, 16.0, 16.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": False
    }
}

hf_model_spec = {
    "gpt-neo-2.7b": {
        "max_memory_per_gpu": [8.0, 8.0, 9.0, 17.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "gpt-j-6b": {
        "max_memory_per_gpu": [8.0, 9.0, 9.0, 21.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "bloom-7b1": {
        "max_memory_per_gpu": [7.0, 7.0, 8.0, 9.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128]
    },
    "bigscience/bloom-3b": {
        "max_memory_per_gpu": [5.0, 6.0],
        "batch_size": [1, 4],
        "seq_length": [16, 32],
        "worker": 1,
        "stream_output": True,
    }
}

ds_model_spec = {
    "gpt-j-6b": {
        "max_memory_per_gpu": [9.0, 10.0, 11.0, 12.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "bloom-7b1": {
        "max_memory_per_gpu": [7.0, 8.0, 8.0, 9.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256]
    },
    "opt-13b": {
        "max_memory_per_gpu": [17.0, 18.0, 19.0, 22.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "worker": 2
    },
    "gpt-neo-1.3b": {
        "max_memory_per_gpu": [4.0, 5.0],
        "batch_size": [1, 4],
        "seq_length": [16],
        "worker": 2,
        "stream_output": True,
    }
}

ft_model_spec = {
    "bigscience/bloom-3b": {
        "batch_size": [1, 2],
        "seq_length": [64, 128],
        "max_memory_per_gpu": [6.0, 6.0, 6.0, 6.0]
    },
    "flan-t5-xxl": {
        "batch_size": [1, 2],
        "seq_length": [64, 128],
        "max_memory_per_gpu": [15.0, 15.0, 15.0, 15.0]
    }
}

ft_raw_model_spec = {
    "t5-small": {
        "batch_size": [1, 2],
        "max_memory_per_gpu": 2
    },
    "gpt2-xl": {
        "batch_size": [1, 2],
        "max_memory_per_gpu": 7.0
    },
    "facebook/opt-6.7b": {
        "batch_size": [1, 2],
        "max_memory_per_gpu": 6.0
    },
    "bigscience/bloom-3b": {
        "batch_size": [1, 2],
        "max_memory_per_gpu": 6.0
    },
    "flan-t5-xxl": {
        "batch_size": [1, 2],
        "max_memory_per_gpu": 15.0
    }
}

sd_model_spec = {
    "stable-diffusion-v1-5": {
        "max_memory_per_gpu": 8.0,
        "size": [256, 512],
        "num_inference_steps": [50, 100]
    },
    "stable-diffusion-2-1-base": {
        "max_memory_per_gpu": 8.0,
        "size": [256, 512],
        "num_inference_steps": [50, 100],
        "workers": 2
    },
    "stable-diffusion-2-depth": {
        "max_memory_per_gpu": 8.0,
        "size": [512],
        "num_inference_steps": [50],
        "depth": True
    }
}

ds_aot_model_spec = {
    "opt-6.7b": {
        "max_memory_per_gpu": [12.0, 12.0, 12.0, 12.0],
        "batch_size": [1, 2, 4, 8],
        "seq_length": [64, 128, 256],
        "use_pipeline": True
    }
}

transformers_neuronx_raw_model_spec = {"gpt2": {"seq_length": [64, 128]}}

transformers_neuronx_model_spec = {
    "opt-1.3b": {
        "worker": 3,
        "seq_length": [128, 256],
        "batch_size": [4]
    },
    "gpt-j-6b": {
        "worker": 1,
        "seq_length": [128, 256, 512],
        "batch_size": [4]
    }
}


def check_worker_number(desired):
    endpoint = "http://127.0.0.1:8080/models/test"
    res = requests.get(endpoint).json()
    if desired == len(res[0]["models"][0]["workerGroups"]):
        return
    elif desired == len(res[0]["models"][0]["workerGroups"][0]["workers"]):
        return
    else:
        raise AssertionError(
            f"Worker number does not meet requirements! {res}")


def send_json(data):
    headers = {'content-type': 'application/json'}
    res = requests.post(endpoint, headers=headers, json=data)
    return res


def send_image_json(img_url, data):
    multipart_form_data = {
        'data': BytesIO(requests.get(img_url, stream=True).content),
        'json': (None, json.dumps(data), 'application/json')
    }
    response = requests.post(endpoint, files=multipart_form_data)
    return response


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]


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


def t5_batch_generation(batch_size):
    input_sentences = [
        "translate English to German: The house is wonderful.",
        "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught \
             of severe weather in mississippi…",
    ]
    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    return input_sentences[:batch_size]


def test_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            req = {"inputs": batch_generation(batch_size)}
            params = {"max_new_tokens": seq_length}
            req["parameters"] = params
            logging.info(f"req {req}")
            res = send_json(req)
            if spec.get("stream_output", False):
                logging.info(f"res: {res.content}")
                result = res.content.decode().split("\n")[:-1]
                assert len(
                    result
                ) <= seq_length, "generated more takens than max_new_tokens"
                result_0 = json.loads(result[0])['outputs']
                assert len(result_0) == batch_size, "batch size number of tokens are not generated"
            else:
                res = res.json()
                logging.info(f"res {res}")

                result = [item['generated_text'] for item in res]
                assert len(result) == batch_size
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"][i]


def test_ds_raw_model(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    for i, batch_size in enumerate(spec["batch_size"]):
        for seq_length in spec["seq_length"]:
            req = {
                "batch_size": batch_size,
                "text_length": seq_length,
                "use_pipeline": spec["use_pipeline"]
            }
            logging.info(f"req: {req}")
            res = send_json(req)
            res = res.json()
            logging.info(f"res: {res}")
            assert len(res["outputs"]) == batch_size
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"][i]


def test_sd_handler(model, model_spec):
    from PIL import Image

    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(sd_model_spec.keys())}"
        )
    spec = sd_model_spec[model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for size in spec["size"]:
        for step in spec["num_inference_steps"]:
            if "depth" in spec:
                req = {"prompt": "two tigers"}
                params = {
                    "negative_prompt": "bad, deformed, ugly, bad anotomy",
                    "strength": 0.7
                }
                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                req["parameters"] = params
                logging.info(f"req: {req}")
                res = send_image_json(url, req)
            else:
                req = {"prompt": "A bird and cat flying through space"}
                params = {
                    "height": size,
                    "width": size,
                    "num_inference_steps": step
                }
                req["parameters"] = params
                logging.info(f"req: {req}")
                res = send_json(req)
            assert res.status_code == 200
            try:
                img = Image.open(BytesIO(res.content)).convert("RGB")
            except Exception as e:
                raise IOError("failed to deserialize image from response", e)
            memory_usage = get_gpu_memory()
            logging.info(memory_usage)
            for memory in memory_usage:
                assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


def test_ft_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(ft_raw_model_spec.keys())}"
        )
    spec = model_spec[model]
    for batch_size in spec['batch_size']:
        logging.info(
            f"testing ft_handler with model: {model}, batch_size: {batch_size} "
        )
        if "t5" in model:
            req = {"inputs": t5_batch_generation(batch_size)}
        else:
            req = {"inputs": batch_generation(batch_size)}
        res = send_json(req)
        res = res.json()
        logging.info(res)
        assert len(res) == batch_size
        memory_usage = get_gpu_memory()
        logging.info(memory_usage)
        for memory in memory_usage:
            assert float(memory) / 1024.0 < spec["max_memory_per_gpu"]


def test_transformers_neuronx_raw(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[model]
    for seq_length in spec["seq_length"]:
        print(
            f"testing transformers_neuronx_handler with model: {model}, seq_length: {seq_length} "
        )
        text = "Hello, I'm a language model,"
        compiled_batch_size = 4
        req = {
            "seq_length": seq_length,
            "text": text,
        }
        res = send_json(req)
        res = res.json()
        logging.info(res)
        assert len(res["outputs"]) == compiled_batch_size
        assert all([text in t for t in res["outputs"]])


def test_transformers_neuronx_handler(model, model_spec):
    if model not in model_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(model_spec.keys())}"
        )
    spec = model_spec[args.model]
    if "worker" in spec:
        check_worker_number(spec["worker"])
    for batch_size in spec["batch_size"]:
        for seq_length in spec["seq_length"]:
            req = {"inputs": batch_generation(batch_size)}
            params = {"max_length": seq_length}
            req["parameters"] = params
            logging.info(f"req {req}")
            res = send_json(req)
            res = res.json()
            logging.info(f"res {res}")
            result = res
            assert len(result) == batch_size


if __name__ == '__main__':
    args = parser.parse_args()
    if args.handler == "deepspeed_raw":
        test_ds_raw_model(args.model, ds_raw_model_spec)
    elif args.handler == "huggingface":
        test_handler(args.model, hf_model_spec)
    elif args.handler == "deepspeed":
        test_handler(args.model, ds_model_spec)
    elif args.handler == "stable-diffusion":
        test_sd_handler(args.model, sd_model_spec)
    elif args.handler == "fastertransformer":
        test_handler(args.model, ft_model_spec)
    elif args.handler == "fastertransformer_raw":
        test_ft_handler(args.model, ft_raw_model_spec)
    elif args.handler == "deepspeed_aot":
        test_ds_raw_model(args.model, ds_aot_model_spec)
    elif args.handler == "transformers_neuronx_raw":
        test_transformers_neuronx_raw(args.model,
                                      transformers_neuronx_raw_model_spec)
    elif args.handler == "transformers_neuronx":
        test_transformers_neuronx_handler(args.model,
                                          transformers_neuronx_model_spec)
    else:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler")
