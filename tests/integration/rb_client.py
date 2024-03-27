import sys
import argparse
import json
import subprocess as sp
import logging

logging.basicConfig(level=logging.INFO)

correctness = {
    "gpt2": {
        "input_texts": [
            "When your legs don'\''t work like they used to before And I can'\''t sweep you off",
            "Memories follow me left and right. I can",
            "The new movie that got Oscar this year",
            "Memories follow me left and right. I can"
        ],
        "parameters": [{
            "max_new_tokens": 40,
            "decoding_strategy": "greedy"
        }, {
            "max_new_tokens": 40,
            "decoding_strategy": "greedy"
        }, {
            "max_new_tokens": 40,
            "decoding_strategy": "contrastive"
        }, {
            "max_new_tokens": 40,
            "decoding_strategy": "contrastive"
        }],
        "expected_outputs": [
            " my feet, I can't do anything about it.\n",
            "'t remember the last time I saw a girl in a dress. I can't remember the last time",
            " is called \"The Dark Knight Rises.\" It's a sequel to Batman v Superman: Dawn of"
            " Justice, which starred Tom Hardy and Laurence Fishburne.",
            "'t remember the last time I saw her.\n\n\"What do you mean?\" asked my mother"
        ]
    }
}

scheduler_single_gpu = {
    "bloom-560m": {
        "input_texts":
        "When your legs don'\''t work like they used to before And I can'\''t sweep you off",
        "parameters": {
            "max_new_tokens": 40,
            "decoding_strategy": "contrastive"
        },
        "expected_word_count": 20,
        "concurrent_clients": 4
    },
    "llama2-7b-chat-gptq": {
        "input_texts":
        "When your legs don'\''t work like they used to before And I can'\''t sweep you off",
        "parameters": {
            "max_new_tokens": 40,
            "decoding_strategy": "greedy"
        },
        "expected_word_count": 20,
        "concurrent_clients": 4
    }
}

scheduler_multi_gpu = {
    "gpt-j-6b": {
        "input_texts":
        "When your legs don'\''t work like they used to before And I can'\''t sweep you off",
        "parameters": {
            "max_new_tokens": 40,
            "decoding_strategy": "contrastive"
        },
        "expected_word_count": 20,
        "concurrent_clients": 4,
    }
}


def send_json(data, output_file, concurrent_clients=1):
    endpoint = f"http://127.0.0.1:8080/invocations"

    commands = [
        "./awscurl", "-X", "POST", endpoint,
        "-H content-type : application/json", "-d",
        str(json.dumps(data)), "-c",
        str(concurrent_clients), "-o", output_file
    ]
    process = sp.Popen(commands)
    return process


def compare_output_in_file(filename, expected_output):
    with open(filename, 'r') as file:
        actual_output = file.read()
        if actual_output.startswith(expected_output):
            return True
        else:
            return False


def count_words_in_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
        word_count = len(content.split())
    return word_count


# sends the multiple requests with same data json concurrently
def test_concurrent_with_same_reqs(model, test_spec, spec_name):
    if model not in test_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(test_spec.keys())}"
        )

    spec = test_spec[model]
    data = {
        "inputs": spec["input_texts"],
        "parameters": spec.get("parameters", {})
    }
    output_file = f"outputs/{model}-{spec_name}-output.txt"

    process = send_json(data=data,
                        output_file=output_file,
                        concurrent_clients=spec.get("concurrent_clients"))
    process.wait()
    if process.returncode != 0:
        raise Exception("Prediction request failed.")

    if "expected_word_count" in spec:
        expected_word_count = spec["expected_word_count"]
        if count_words_in_file(output_file) < expected_word_count:
            raise AssertionError(
                "Did not produce the expected number of words")


# sends multiple requests with different data json concurrently
def test_concurrent_with_mul_reqs(model, test_spec, spec_name):
    if model not in test_spec:
        raise ValueError(
            f"{args.model} is not one of the supporting models {list(test_spec.keys())}"
        )

    processes = []
    spec = test_spec[model]
    for index in range(len(spec.get("input_texts"))):
        input_text = spec["input_texts"][index]
        parameter = spec.get("parameters", [{}])[index]
        data = {"inputs": input_text, "parameters": parameter}
        process = send_json(data,
                            f"outputs/{model}-{spec_name}-output{index}.txt")
        processes.append(process)

    # waiting for each curl command to finish
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise Exception("Prediction request failed.")

    if "expected_outputs" in spec:
        for index, expected_output in enumerate(spec["expected_outputs"]):
            output_file = f"outputs/{model}-{spec_name}-output{index}.txt"
            if not compare_output_in_file(output_file, expected_output):
                return AssertionError("Expected output was not matched.")

    if "expected_word_count" in spec:
        for index, expected_word_count in enumerate(
                spec["expected_word_count"]):
            output_file = f"outputs/{model}-output{index}.txt"
            if count_words_in_file(output_file) < expected_word_count:
                raise AssertionError(
                    "Did not produce the expected number of words")


def run(raw_args):
    parser = argparse.ArgumentParser(
        description="Build the rolling batch configs")
    parser.add_argument("test_spec", help="type of test that needs to be run")
    parser.add_argument("model", help="The name of model")
    global args
    args = parser.parse_args(args=raw_args)

    if args.test_spec == "correctness":
        test_concurrent_with_mul_reqs(args.model, correctness, "correctness")
    elif args.test_spec == "scheduler_single_gpu":
        test_concurrent_with_same_reqs(args.model, scheduler_single_gpu,
                                       "scheduler_single_gpu")
    elif args.test_spec == "scheduler_multi_gpu":
        test_concurrent_with_same_reqs(args.model, scheduler_multi_gpu,
                                       "scheduler_multi_gpu")
    else:
        raise ValueError(
            f"{args.handler} is not one of the supporting handler")


if __name__ == "__main__":
    run(sys.argv)
