from pathlib import Path
from datetime import datetime
from run_benchmark import run_benchmark_job
import argparse
import boto3
import csv
import logging
import os
import yaml


def parse_csv_table(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        field_names = next(reader)
        data_list = []
        for row in reader:
            row_dict = {}
            for i, value in enumerate(row):
                row_dict[field_names[i]] = value
            data_list.append(row_dict)

    return data_list


def publish_cw_metrics(
    cw, metrics_namespace, metrics, csv_table, model_name, endpoint_type, container
):
    for row in csv_table:
        for field_name, field_value in row.items():
            if field_name in metrics:
                if (
                    field_name == "tokenizerFailed_Sum"
                    or field_name == "emptyInferenceResponse_Sum"
                    or field_name == "clientInvocationErrors_Sum"
                ):
                    if float(row["clientInvocations_Sum"]) > 0.0:
                        field_value = (
                            float(row[field_name])
                            * 100
                            // float(row["clientInvocations_Sum"])
                        )
                    else:
                        field_value = 0.0

                metric_data = [
                    {
                        "MetricName": metrics[field_name]["metric_name"],
                        "Dimensions": [
                            {
                                "Name": "Model",
                                "Value": model_name,
                            },
                            {
                                "Name": "Endpoint",
                                "Value": endpoint_type,
                            },
                            {
                                "Name": "Container",
                                "Value": container,
                            },
                            {
                                "Name": "InstanceType",
                                "Value": row["InstanceType"],
                            },
                            {
                                "Name": "Concurrency",
                                "Value": row["Concurrency"],
                            },
                        ],
                        "Unit": metrics[field_name]["unit"],
                        "Value": float(field_value),
                    }
                ]
                response = cw.put_metric_data(
                    Namespace=metrics_namespace, MetricData=metric_data
                )
                logging.info(
                    "publish metric: %s, model: %s, endpoint: %s, container: %s, instance_type: %s, concurrency: %s, response: %s",
                    metrics[field_name]["metric_name"],
                    model_name,
                    endpoint_type,
                    container,
                    row["InstanceType"],
                    row["Concurrency"],
                    response,
                )


def run_benchmark(config_yml, benchmark_config_dir, benchmark_metric_dir):
    with open(config_yml, "r") as file:
        config = yaml.safe_load(file)

        if config is None:
            logging.fatal("Invalid config.yml")
        region = config.get("region", "us-west-2")
        metrics_namespace = config.get("cloudwatch", {}).get(
            "metrics_namespace", "Rubikon"
        )
        metrics = config.get("metrics", {})
        hf_token = os.getenv("HF_TOKEN", "")
        s3_bucket = config.get("s3", {}).get("bucket_name", "djl-benchmark")
        s3_folder = config.get("s3", {}).get("folder", "lmi-dist")
        current_date = datetime.now().strftime("%Y-%m-%d")
        s3_metrics_folder = f"{current_date}/{s3_folder}/metrics/"
        s3_config_folder = f"{current_date}/{s3_folder}/config/"
        s3_job_config_object = f"{s3_config_folder}config.yml"

        for benchmark in config["benchmarks"]:
            model = benchmark["model"]
            endpoints = benchmark["endpoints"]
            for ep in endpoints:
                action = ep["action"]
                if not action:
                    continue
                endpoint = ep["endpoint"]
                image = ep["image"]
                config_file = ep["config"]
                dataset = ep["dataset"]
                try:
                    run_benchmark_job(
                        Path(benchmark_config_dir).joinpath(config_file),
                        dataset,
                        benchmark_metric_dir,
                        region,
                        hf_token,
                    )
                    csv_file = config_file[:-4] + "csv"
                    s3_metrics_object = f"{s3_metrics_folder}{csv_file}"
                    csv_file = Path(benchmark_metric_dir).joinpath(csv_file)
                    csv_table = parse_csv_table(csv_file)

                    s3_config_object = f"{s3_config_folder}{config_file}"
                    session = boto3.session.Session()
                    cloudwatch = session.client("cloudwatch", region_name=region)
                    s3 = session.client("s3", region_name=region)
                
                    s3.upload_file(
                        Path(config_yml),
                        s3_bucket,
                        s3_job_config_object,
                    )
                    s3.upload_file(
                        Path(benchmark_config_dir).joinpath(config_file),
                        s3_bucket,
                        s3_config_object,
                    )
                    logging.info(
                        f"file {config_file} uploaded successfully to s3://{s3_bucket}/{s3_config_object}"
                    )
                    s3.upload_file(csv_file, s3_bucket, s3_metrics_object)
                    logging.info(
                        f"file {csv_file} uploaded successfully to s3://{s3_bucket}/{s3_metrics_object}"
                    )
                
                    publish_cw_metrics(
                        cloudwatch,
                        metrics_namespace,
                        metrics,
                        csv_table,
                        model,
                        endpoint,
                        image,
                    )
                except Exception as e:
                    logging.error(f"Error in running benchmark with {config_file} on {dataset}, error: {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", "-j", type=str, required=True, help="Specify the config.yml path"
    )
    parser.add_argument(
        "--configs",
        "-c",
        type=Path,
        required=True,
        help="Specify the benchmark config path",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=Path,
        required=True,
        help="Specify the benchmark metrics path",
    )
    args = parser.parse_args()

    run_benchmark(args.jobs, args.configs, args.metrics)


if __name__ == "__main__":
    main()
