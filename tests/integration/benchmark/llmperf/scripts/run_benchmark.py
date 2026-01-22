from ast import Str
from pathlib import Path
from datetime import datetime
import string
import subprocess
import argparse
import boto3
import json
import logging
import os
import psutil
import shlex
import tempfile
import time
import yaml
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(filename='run_benchmark.log',
                    encoding='utf-8',
                    level=logging.INFO)


def publish_cw_metrics(cw, metrics_namespace, metrics, json_map, model_name,
                       endpoint_type, container, instance_type, concurrency):
    for field_name, field_value in json_map.items():
        if field_name in metrics:
            metric_data = [{
                "MetricName":
                metrics[field_name]["metric_name"],
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
                        "Value": instance_type,
                    },
                    {
                        "Name": "Concurrency",
                        "Value": concurrency,
                    },
                ],
                "Unit":
                metrics[field_name]["unit"],
                "Value":
                float(field_value),
            }]
            response = cw.put_metric_data(Namespace=metrics_namespace,
                                          MetricData=metric_data)
            logger.info(
                "publish metric: %s, model: %s, endpoint: %s, container: %s, instance_type: %s, concurrency: %s, response: %s",
                metrics[field_name]["metric_name"],
                model_name,
                endpoint_type,
                container,
                instance_type,
                concurrency,
                response,
            )


def is_valid_device(devices, ec2_instance_type):
    for device in devices:
        if device == ec2_instance_type:
            return True
    return False


def run_bash_command(bash_command):
    try:
        logger.info(f"{bash_command}")
        output = subprocess.run(bash_command,
                                capture_output=True,
                                text=True,
                                check=True)

        for line in output.stdout.splitlines():
            logger.info(line)

        for line in output.stderr.splitlines():
            logger.error(line)

        return output.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {bash_command}, error: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Not found command: {bash_command}")
        return None


def pull_docker_image(container):
    try:
        image_name = get_image_name(container)
        logger.info(f"image_name={image_name}")
    except ValueError as e:
        return None
    bash_command = f"docker pull {image_name}"
    if run_bash_command(bash_command.split()) is None:
        return None
    return image_name


def build_parameter_str(parameter_list):
    if len(parameter_list) == 0:
        return ""
    return " ".join(parameter_list)


def launch_container(container, image, model, hf_token,
                     container_log_file_name):
    docker_parameters = build_parameter_str(
        container.get("docker_parameters", []))
    server_parameters = build_parameter_str(
        container.get("server_parameters", []))

    try:
        container_name = container["container"]
        docker_command = f"docker run --name {container_name} --rm -e HUGGING_FACE_HUB_TOKEN={hf_token} {docker_parameters} --init {image} {server_parameters}"
        logger.info(f"docker_command={docker_command}")
        # Open log file for writing
        log_file = open(Path(container_log_file_name), "w")

        process = subprocess.Popen(docker_command,
                                   shell=True,
                                   stdout=log_file,
                                   stderr=subprocess.STDOUT)
        logger.info(f"Docker run shell process={process.pid}")
        time.sleep(60)
        container = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.Name}}", container_name])
    except Exception as e:
        logger.error(f"Failed to launch container. error: {e}")
        logger.error(traceback.format_exc())
        raise e


def wait_for_server(model, timeout=7200):
    # wait for server to start return 1 if  server crashes
    script = '''
    #!/bin/bash
    MODEL_ID=$1
    TIMEOUT=$2
    start_time=$(date +%s)
    end_time=$((start_time + $TIMEOUT)) 
    while [ $(date +%s) -lt $end_time ]; do
        filter="$(curl -s http://0.0.0.0:8080/v1/models |jq -e --arg expected "$MODEL_ID" '. != null and . != {} and has("data") and .data != null and (.data | length > 0) and (.data[].id | contains($expected))')"
        if [ -z "$filter" ]; then
            echo "Model $MODEL_ID is not available"
            sleep 1m
        else
            echo "Model $MODEL_ID is available"
            exit 0
        fi
    done

    echo "Model $MODEL_ID is not available within 2 hours"
    exit 1
    '''
    bash_command = ['bash', '-c', script, 'my_script', model, str(timeout)]
    if run_bash_command(bash_command) is None:
        logger.error(traceback.format_exc())
        raise Exception("Failed at starting server within 2 hours")


def load_json_to_map(file_path):
    data_map = {}
    try:
        with open(file_path, 'r') as file:
            data_map = json.load(file)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in the file. error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred. error: {e}")
    if (not bool(data_map)):
        raise Exception(
            "Failed at load_json_to_map for loading llmperf benchmark summary")
    return data_map


def run_llm_perf(warmup: bool, model, concurrency,
                 llmperf_parameters_others_list, result_outputs, llmperf_path):
    if warmup:
        bash_command = f"python {llmperf_path}/token_benchmark_ray.py --model {model} --num-concurrent-requests 2 --mean-input-tokens 550 --stddev-input-tokens 1150 --mean-output-tokens 150 --stddev-output-tokens 10 --max-num-completed-requests 50 --timeout 600 \
--num-concurrent-requests 1 --llm-api openai"

        if run_bash_command(bash_command.split()) is None:
            raise Exception("Failded at warmup")
        return

    llmperf_parameters_others = build_parameter_str(
        llmperf_parameters_others_list)
    bash_command = f"python {llmperf_path}/token_benchmark_ray.py --model {model} --results-dir {result_outputs} --num-concurrent-requests {concurrency} {llmperf_parameters_others}"
    if run_bash_command(bash_command.split()) is None:
        raise Exception("Failded at run_llm_perf")
    return


def shutdown_container(container_name):
    bash_command = f"docker rm -f {container_name}"
    if run_bash_command(bash_command.split()) is None:
        logger.error(traceback.format_exc())


def delete_docker_images():
    bash_command = "docker rmi $(docker images -a -q) -f"
    if run_bash_command(bash_command.split()) is None:
        raise Exception("Failded at delete_docker_images")


def upload_summary_to_s3(s3, result_outputs, s3_bucket, s3_metrics_folder):
    summary_json_file = None
    s3_metrics_object = None
    for file in Path(result_outputs).iterdir():
        if file.is_file():
            try:
                if file.name.endswith("_summary.json"):
                    summary_json_file = Path(file)
                s3_metrics_object = f"{s3_metrics_folder}{file}"
                logger.info(
                    f"upload to s3_metrics_object: {s3_metrics_object}")
                s3.upload_file(Path(file), s3_bucket, s3_metrics_object)
            except Exception as e:
                logger.error(
                    f'Failed to upload {Path(file)} to {s3_metrics_object}, error: {e}'
                )
    if summary_json_file is None:
        raise Exception("Failed at upload_summary_to_s3")

    return summary_json_file


def get_ec2_instance_type():
    script = '''
    #!/bin/bash
    INSTANCE_TYPE=$(TOKEN=`curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` \
&& curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)

    if [ -n "$INSTANCE_TYPE" ]; then
        echo "$INSTANCE_TYPE"
        exit 0
    else
        echo "No instance type found." >&2
        exit 1
    fi
    '''
    bash_command = ['bash', '-c', script]
    return run_bash_command(bash_command)


def get_public_ecr_image_latest_tag(repo):
    # ecr login via workflow
    # aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
    script = '''
    #!/bin/bash
    REPO="$1"
    LATEST_TAG=$(TOKEN=$(curl -s -k https://public.ecr.aws/token/ | jq -r '.token') &&  curl -s -k -H "Authorization: Bearer $TOKEN"  https://public.ecr.aws/v2/$REPO/tags/list | jq -r '.tags[0] // empty')
    
    if [ -n "$LATEST_TAG" ]; then
        echo "$LATEST_TAG"
        exit 0
    else
        echo "No tags found or unable to parse the JSON response." >&2
        exit 1
    fi
    '''
    bash_command = ['bash', '-c', script, 'my_script', repo]
    tag = run_bash_command(bash_command)
    if tag:
        return f'public.ecr.aws/{repo}:{tag}'
    return None


def get_vllm_image(repo, tag=None):
    #repo default public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo
    if tag is None:
        image = get_public_ecr_image_latest_tag(
            "q9t5s3a7/vllm-ci-postmerge-repo")
        if image is None:
            raise ValueError("Invalid vllm repo: {repo} in public.ecr.aws")
    else:
        image = f"{repo}:{tag}"
    return image


def get_public_ghcr_image_latest_tag(repo):
    #https://github.com/orgs/community/discussions/26279
    #export GHCR_TOKEN=$(echo $GITHUB_TOKEN | base64)
    script = '''
    #!/bin/bash
    REPO="$1"
    LATEST_TAG=$(curl -s -k -H "Authorization: Bearer $GHCR_TOKEN" https://ghcr.io/v2//$REPO/tags/list | jq -r '.tags[0] // empty')
    
    if [ -n "$LATEST_TAG" ]; then
        echo "$LATEST_TAG"
        exit 0
    else
        echo "No tags found or unable to parse the JSON response." >&2
        exit 1
    fi
    '''
    bash_command = ['bash', '-c', script, 'my_script', repo]
    tag = run_bash_command(bash_command)
    if tag:
        return f'ghcr.io/{repo}:{tag}'
    return None


def get_tgi_image(repo, tag=None):
    #repo default ghcr.io/huggingface/text-generation-inference
    if tag is None:
        image = get_public_ghcr_image_latest_tag(
            "huggingface/text-generation-inference")
        if image is None:
            raise ValueError("Invalid vllm repo: {repo} in ghcr.io")
    else:
        image = f"{repo}:{tag}"
    return image


def get_public_ngc_image_latest_tag(repo):
    script = '''
    #!/bin/bash
    REPO="$1"
    LATEST_TAG=$(./ngc registry image list $REPO --format_type json | jq 'sort_by(.updatedDate) | reverse | [.[].tag | select(contains("trtllm"))] | first // empty')
    
    if [ -n "$LATEST_TAG" ]; then
        echo "$LATEST_TAG"
        exit 0
    else
        echo "No tags found or unable to parse the JSON response." >&2
        exit 1
    fi
    '''
    bash_command = ['bash', '-c', script, 'my_script', repo]
    tag = run_bash_command(bash_command)
    if tag:
        return f'{repo}:{tag}'
    return None


def get_trtllm_image(repo, tag):
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
    # https://github.com/triton-inference-server/server/blob/main/README.md
    #repo default nvcr.io/nvidia/tritonserver
    # tag 24.07-trtllm-python-py3
    if tag is None:
        image = get_public_ngc_image_latest_tag("nvcr.io/nvidia/tritonserver")
        if image is None:
            raise ValueError("Invalid vllm repo: {repo} in ghcr.io")
    else:
        image = f"{repo}:{tag}"
    return image


def get_image_name(container):
    container_name = container.get("container")
    if container_name == "vllm":
        repo = container.get("repo",
                             "public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo")
        tag = container.get("tag", None)
        return get_vllm_image(repo, tag)
    elif container_name == "tgi":
        repo = container.get("repo",
                             "ghcr.io/huggingface/text-generation-inference")
        tag = container.get("tag", None)
        return get_tgi_image(repo, tag)
    elif container_name == "trtllm":
        repo = container.get("repo", "nvcr.io/nvidia/tritonserver")
        tag = container.get("tag", None)
        return get_trtllm_image(repo, tag)

    repo = container.get("repo", None)
    tag = container.get("tag", None)
    if repo is None or tag is None:
        raise ValueError(
            "Not found repo and tag in container: {container_name}")
    return f"{repo}:{tag}"


def run_benchmark(config_yml, llmperf_path):
    os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
    os.environ["OPENAI_API_KEY"] = "EMPTY"

    ec2_instance_type = get_ec2_instance_type()
    if ec2_instance_type is None:
        logger.fatal("Failed to get EC2 instance type")
        return

    with open(config_yml, "r") as file:
        config = yaml.safe_load(file)

        if config is None:
            logger.fatal("Invalid config.yml")
        region = config.get("region", "us-west-2")
        metrics_namespace = config.get("cloudwatch",
                                       {}).get("metrics_namespace", "Rubikon")
        metrics = config.get("metrics", {})
        hf_token = os.getenv("HF_TOKEN", "")
        s3_bucket = config.get("s3", {}).get("bucket_name",
                                             "djl-benchmark-llm")
        s3_folder = config.get("s3", {}).get("folder", "ec2")
        current_date = datetime.now().strftime("%Y-%m-%d")
        s3_metrics_folder = f"{current_date}/{s3_folder}/metrics/"
        s3_config_folder = f"{current_date}/{s3_folder}/config/"
        s3_job_config_object = f"{s3_config_folder}config.yml"
        session = boto3.session.Session()
        cloudwatch = session.client("cloudwatch", region_name=region)
        s3 = session.client("s3", region_name=region)
        s3.upload_file(Path(config_yml), s3_bucket, s3_job_config_object)

        for benchmark in config["benchmarks"]:
            model = benchmark["model"]
            tests = benchmark["tests"]
            for test in tests:
                test_name = test.get("test_name")
                containers = test.get("containers")
                llmperf_parameters = test.get("llmperf_parameters")

                for container in containers:
                    if not container["action"]:
                        continue

                    if not is_valid_device(container["instance_types"],
                                           ec2_instance_type):
                        continue
                    logger.info(f"ec2_instance_type:{ec2_instance_type}")

                    image = pull_docker_image(container)
                    if image is None:
                        continue
                    logger.info(f"image={image}")

                    container_name = container.get("container")
                    timeout = container.get("timeout", 7200)
                    for concurrency in llmperf_parameters[
                            "num-concurrent-requests-list"]:
                        try:
                            logger.info(
                                f"start {model}-{test_name} on container {container_name} with concurrency {concurrency}"
                            )
                            container_log_file_name = f"{model}_{test_name}_{container_name}_{concurrency}.log".replace(
                                "/", "_")
                            launch_container(container, image, model, hf_token,
                                             container_log_file_name)
                            wait_for_server(model, timeout)
                            logger.info("server started successfully")

                            result_outputs = f"{container_name}/{model}-{test_name}/{concurrency}"

                            logger.info("start warmup")
                            run_llm_perf(True, model, concurrency,
                                         llmperf_parameters.get("others", []),
                                         result_outputs, llmperf_path)
                            logger.info("start llmperf")
                            run_llm_perf(False, model, concurrency,
                                         llmperf_parameters.get("others", []),
                                         result_outputs, llmperf_path)
                            logger.info("shutdowm container")
                            shutdown_container(container_name)

                            # parse llmperf json file
                            # upload to s3
                            # publish metrics
                            logger.info("upload llmperf summary to s3")
                            summary_json_file = upload_summary_to_s3(
                                s3, result_outputs, s3_bucket,
                                s3_metrics_folder)
                            json_map = load_json_to_map(summary_json_file)
                            logger.info("upload metrics to cloudwatch")
                            publish_cw_metrics(cloudwatch, metrics_namespace,
                                               metrics, json_map, model, "ec2",
                                               container_name,
                                               ec2_instance_type,
                                               str(concurrency))
                        except Exception as e:
                            logger.error(
                                f'Error in test: {test["test_name"]} on {container["container"]} with concurrency {concurrency}, error: {e}'
                            )
    #delete_docker_images()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",
                        "-j",
                        type=str,
                        required=True,
                        help="Specify the config.yml path")
    parser.add_argument("--path",
                        "-p",
                        type=Path,
                        required=True,
                        help="Specify the llmperf path")

    args = parser.parse_args()

    run_benchmark(args.jobs, args.path)


if __name__ == "__main__":
    main()
