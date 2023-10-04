import argparse
import logging
import json
import os
import shutil
import urllib.request
import subprocess as sp

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Script for building and running the LMI container tests")
parser.add_argument("--parse",
                    required=False,
                    type=str,
                    help="Parse the template")
parser.add_argument("--template",
                    required=False,
                    type=str,
                    help="The template json string")
parser.add_argument("--job", required=False, type=str, help="The job string")
args = parser.parse_args()


def is_square_bracket(input_str):
    return input_str[0] == '[' and input_str[-1] == ']'


def parse_raw_template(url):
    data = urllib.request.urlopen(url)
    lines = [line.decode("utf-8").strip() for line in data]
    iterator = 0
    final_result = {}
    name = ''
    properties = []
    commandline = []
    requirements = []
    while iterator < len(lines):
        if '[test_name]' == lines[iterator]:
            iterator += 1
            name = lines[iterator]
        elif '[serving_properties]' == lines[iterator]:
            iterator += 1
            while iterator < len(lines) and not is_square_bracket(
                    lines[iterator]):
                properties.append(lines[iterator])
                iterator += 1
        elif '[requirements]' == lines[iterator]:
            iterator += 1
            while iterator < len(lines) and not is_square_bracket(
                    lines[iterator]):
                requirements.append(lines[iterator])
                iterator += 1
        elif '[aws_curl]' == lines[iterator]:
            iterator += 1
            while iterator < len(lines) and not is_square_bracket(
                    lines[iterator]):
                commandline.append(lines[iterator])
                iterator += 1
        else:
            iterator += 1
        if name and properties and commandline:
            final_result[name] = {
                "properties": properties,
                "awscurl": ' '.join(commandline),
                "requirements": requirements
            }
            name = ''
            properties = []
            commandline = []
            requirements = []
    return final_result


def write_model_artifacts(properties, requirements=None):
    model_path = "models/test"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "serving.properties"), "w") as f:
        f.write('\n'.join(properties) + '\n')
    if requirements:
        with open(os.path.join(model_path, "requirements.txt"), "w") as f:
            f.write('\n'.join(requirements) + '\n')


def build_running_script(template, job):
    template = json.loads(template)
    job_template = template[job]
    write_model_artifacts(job_template['properties'],
                          job_template['requirements'])


if __name__ == "__main__":
    if args.parse_template:
        result = parse_raw_template(args.parse_template)
        command = f"echo \"::set-output name=jobs::{json.dumps(result.keys())}\""
        sp.call(command, shell=True)
        command = f"echo \"::set-output name=template::{json.dumps(result)}\""
        sp.call(command, shell=True)
    elif args.template and args.job:
        build_running_script(args.template, args.job)
    else:
        parser.print_help()
        raise ValueError("args not supported")
