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
parser.add_argument("--container",
                    required=False,
                    type=str,
                    help="The container to run the job")
parser.add_argument("--instance",
                    required=False,
                    type=str,
                    help="The current instance name")

parser.add_argument("--job", required=False, type=str, help="The job string")
args = parser.parse_args()


def is_square_bracket(input_str):
    return input_str[0] == '[' and input_str[-1] == ']'


def parse_raw_template(url):
    data = urllib.request.urlopen(url)
    lines = [line.decode("utf-8").strip() for line in data]
    # remove empty lines
    lines = [line for line in lines if len(line) > 0]
    iterator = 0
    final_result = {}
    name = ''
    properties = []
    commandline = []
    requirements = []
    info = None
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
                commandline.append(lines[iterator].replace("\\", " "))
                iterator += 1
        elif '[info]' == lines[iterator]:
            info = []
            iterator += 1
            while iterator < len(lines) and not is_square_bracket(
                    lines[iterator]):
                info.append(lines[iterator].replace("\\", " "))
                iterator += 1
        else:
            iterator += 1
        if name and properties and commandline:
            cur_result = {
                "properties": properties,
                "awscurl": ' '.join(commandline).encode().hex(),
                "requirements": requirements
            }
            if info is not None:
                cur_result['info'] = info
            final_result[name] = cur_result
            name = ''
            properties = []
            commandline = []
            requirements = []
            info = None
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


def machine_translation(machine_name: str):
    translation = {
        "inf2.2xlarge": 1,
        "inf2.8xlarge": 1,
        "inf2.24xlarge": 6,
        "inf2.48xlarge": 12,
        "trn1.2xlarge": 1,
        "trn1.32xlarge": 16
    }
    if machine_name.startswith("inf2") or machine_name.startswith("trn1"):
        return f"pytorch-inf2-{translation[machine_name]}"
    else:
        return "deepspeed"


def build_running_script(template, job, instance, container):
    with open(template) as f:
        template = json.load(f)
    job_template = template[job]
    job_template['awscurl'] = bytes.fromhex(
        job_template['awscurl']).decode("utf-8")
    write_model_artifacts(job_template['properties'],
                          job_template['requirements'])

    command_str = f"./launch_container.sh {container} $PWD/models {machine_translation(instance)}"
    bash_command = [
        'echo "Start Launching container..."', command_str,
        job_template['awscurl'] + " | tee benchmark.log"
    ]
    with open("instant_benchmark.sh", "w") as f:
        f.write('\n'.join(bash_command))


if __name__ == "__main__":
    if args.parse:
        result = parse_raw_template(args.parse)
        logging.info(f"Parsed running instruction: {result}")
        command = f"echo \"jobs={json.dumps(json.dumps(list(result.keys())))}\" >> $GITHUB_OUTPUT"
        sp.call(command, shell=True)
        command = f"echo \"template={json.dumps(json.dumps(json.dumps(result)))}\" >> $GITHUB_OUTPUT"
        sp.call(command, shell=True)
    elif args.template and args.job and args.instance and args.container:
        build_running_script(args.template, args.job, args.instance,
                             args.container)
    else:
        parser.print_help()
        raise ValueError("args not supported")
