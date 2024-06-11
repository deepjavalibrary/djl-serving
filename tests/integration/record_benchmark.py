#!/usr/bin/env python3

import argparse
import logging
import boto3
import json
import os
import time
from decimal import Decimal

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Script for saving container benchmark results")
parser.add_argument("--container",
                    required=False,
                    type=str,
                    help="The container to run the job")
parser.add_argument("--template",
                    required=False,
                    type=str,
                    help="The template json string")
parser.add_argument("--job",
                    required=False,
                    type=str,
                    help="The job item within the template")
parser.add_argument("--instance",
                    required=True,
                    type=str,
                    help="The current instance name")
parser.add_argument("--record",
                    choices=["table", "cloudwatch"],
                    required=False,
                    type=str,
                    help="Where to record to")
parser.add_argument("--model",
                    required=False,
                    type=str,
                    help="The path to the model input directory")
parser.add_argument("--benchmark-vars",
                    required=False,
                    type=str,
                    help="The benchmark variables used to differentiate in"
                    " cloudwatch like [CONCURRENCY=2,DATASET=gsm8k]")
parser.add_argument("--info",
                    required=False,
                    type=str,
                    nargs="+",
                    help="A set of info in format of --info a=1 b=2 c=3")
args = parser.parse_args()

cloudwatch_report_schema = {
    "totalTimeMills": 'Milliseconds',
    "totalRequests": 'Count',
    "failedRequests": 'Count',
    "concurrentClients": 'Count',
    "totalTokens": 'Count',
    "tokenPerRequest": 'Count',
    "averageLatency": 'Milliseconds',
    "p50Latency": 'Milliseconds',
    "p90Latency": 'Milliseconds',
    "p99Latency": 'Milliseconds',
    "timeToFirstByte": 'Milliseconds',
    "p50TimeToFirstByte": 'Milliseconds',
    "p90TimeToFirstByte": 'Milliseconds',
    "p99TimeToFirstByte": 'Milliseconds',
}


class Benchmark:

    def __init__(self, dyn_resource, data: dict):
        self.dyn_resource = dyn_resource
        self.table = dyn_resource.Table("RubikonBenchmarks")
        self.table.load()
        self.data = data

    def add_benchmark(self):
        self.table.put_item(Item=self.data)


def record_table(data: dict):
    table = boto3.resource("dynamodb").Table("RubikonBenchmarks")
    table.put_item(Item=data)


def record_cloudwatch(data: dict):
    esc = lambda n: n.replace("/", "-").replace(".", "-").replace("=", "-"
                                                                  ).strip(' -')
    job_name = data["modelId"] if "job" not in data else data["job"]
    benchmark_vars = data["benchmark_vars"] if data["benchmark_vars"] else ""
    metric_name = lambda n: (f"lmi_{data['instance']}_{esc(data['image'])}"
                             f"_{esc(job_name)}_{esc(benchmark_vars)}_{n}")
    metric_data = []
    for metric, unit in cloudwatch_report_schema.items():
        if metric in data.keys():
            metric_data.append({
                'MetricName': metric_name(metric),
                'Unit': unit,
                'Value': data[metric]
            })
    cw = boto3.client('cloudwatch', region_name='us-east-1')
    cw.put_metric_data(Namespace="LMI_Benchmark", MetricData=metric_data)


def data_basic(data: dict):
    data["modelServer"] = "DJLServing"
    data["service"] = "ec2"
    data["Timestamp"] = Decimal(time.time())

    data["instance"] = args.instance

    if args.container is not None:
        data["container"] = args.container

    if args.info:
        for info in args.info:
            split = info.split("=", 1)
            data[[split[0]]] = split[1]


def data_from_client(data: dict):
    if os.path.exists("benchmark_result.json"):
        with open("benchmark_result.json", "r") as f:
            benchmark_metrics = json.load(f)
            data.update(benchmark_metrics)
            print(f"found awscurl metrics json: {benchmark_metrics}")
    elif os.path.exists("benchmark.log"):
        with open("benchmark.log", "r") as f:
            for line in f.readlines():
                line = line.strip()
                if "Total time:" in line:
                    data["totalTimeMills"] = Decimal(line.split(" ")[2])
                if "error rate:" in line:
                    data["errorRate"] = Decimal(line.split(" ")[-1])
                if "Concurrent clients:" in line:
                    data["concurrentClients"] = int(line.split(" ")[2])
                if "Total requests:" in line:
                    data["totalRequests"] = int(line.split(" ")[2])
                if "TPS:" in line:
                    data["tps"] = Decimal(line.split(" ")[1].split("/")[0])
                if "Average Latency:" in line:
                    data["averageLatency"] = Decimal(line.split(" ")[2])
                if "P50:" in line:
                    data["p50Latency"] = Decimal(line.split(" ")[1])
                if "P90:" in line:
                    data["p90Latency"] = Decimal(line.split(" ")[1])
                if "P99:" in line:
                    data["p99Latency"] = Decimal(line.split(" ")[1])
    else:
        print("There is no benchmark logs found!")


def data_container(data: dict):
    if "container" in data:
        container = data["container"]
        if container.startswith("deepjavalibrary/djl-serving:"):
            container = container[len("deepjavalibrary/djl-serving:"):]
            if container[0] == "0":  # Release build
                split = container.split("-", 1)
                data["djlVersion"] = split[0]
                if len(split) > 1:
                    data["image"] = split[1]
                else:
                    data["image"] = "cpu"
            else:  # Nightly build
                data["djlNightly"] = "true"
                data["image"] = container[:-len("-nightly")]
        elif "text-generation-inference" in container:
            data["modelServer"] = "TGI"
            version = container.split(":")[1]
            if not version.startswith("sha"):
                data["tgiVersion"] = version
        elif "suzuka" in container:
            data["image"] = "suzuka"
        else:
            data["image"] = "other"


def data_from_model_files(data: dict):
    if args.model:
        propsPath = os.path.join(args.model, "serving.properties")
        if os.path.isfile(propsPath):
            with open(propsPath, "r") as f:
                properties = {}
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if "=" in line:
                        split = line.split("=", 1)
                        k = split[0].replace(".", "-")
                        v = split[1].replace(".", "-")
                        properties[k] = v
                data["serving_properties"] = properties

                # Standard properties
                if "option-model_id" in properties:
                    data["modelId"] = properties["option-model_id"]
                if "option-tensor_parallel_degree" in properties:
                    data["tensorParallel"] = properties[
                        "option-tensor_parallel_degree"]

        requirementsPath = os.path.join(args.model, "requirements.txt")
        if os.path.isfile(requirementsPath):
            with open(requirementsPath, "r") as f:
                req = {}
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if "=" in line:
                        split = line.split("=", 1)
                        req[split[0]] = split[1]
                data["requirements_txt"] = req

        envPath = os.path.join(args.model, "docker_env")
        if os.path.isfile(envPath):
            with open(envPath, "r") as f:
                envs = {}
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if "=" in line:
                        split = line.split("=", 1)
                        envs[split[0]] = split[1]
                data["env"] = envs
                if "modelId" not in data and "MODEL_ID" in envs:
                    data["modelId"] = envs["MODEL_ID"]


def data_from_template(data: dict):
    if args.template:
        with open(args.template, "r") as f:
            template = json.load(f)
            job_template = template[args.job]
            data["job"] = args.job
            data['benchmark_vars'] = args.benchmark_vars
            data["awscurl"] = bytes.fromhex(
                job_template['awscurl'][args.benchmark_vars]).decode("utf-8")
            if "container" not in data and "container" in job_template:
                data["container"] = job_template["container"]
            if "info" in job_template:
                for line in job_template["info"]:
                    split = line.split("=", 1)
                    data[split[0]] = split[1]
                return True
            else:
                return False


if __name__ == "__main__":
    data = {}
    data_from_template(data)
    data_basic(data)
    data_container(data)
    data_from_client(data)
    data_from_model_files(data)

    if "errorRate" not in data or data["errorRate"] == 100:
        print("Not recording failed benchmark")
        print(data)
    else:
        if args.record == "table":
            record_table(data)
        elif args.record == "cloudwatch":
            record_cloudwatch(data)
        else:
            print(data)
