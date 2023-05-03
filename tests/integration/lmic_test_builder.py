import argparse
import logging
import json
import os
import re
import subprocess as sp
from itertools import product

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Script for building and running the LMI container tests")
parser.add_argument("--docker_image",
                    required=True,
                    type=str,
                    help="Docker image under test")
parser.add_argument("--test_series",
                    required=False,
                    type=str,
                    help="Test series used for testing the model")
parser.add_argument("--model",
                    required=False,
                    type=str,
                    help="The name of test model")
parser.add_argument("--engine",
                    required=False,
                    type=str,
                    choices=["deepspeed", "huggingface", "fastertransformer"],
                    help="The engine used for inference")
parser.add_argument("--platform",
                    default="deepspeed",
                    required=False,
                    type=str,
                    help="The model data type")
parser.add_argument("--dtype",
                    required=False,
                    type=str,
                    help="The model data type")
parser.add_argument("--tensor_parallel",
                    required=False,
                    type=int,
                    help="The model tensor parallel degree")
parser.add_argument("--batch_size",
                    required=False,
                    type=int,
                    help="The batch size of inference requests")
parser.add_argument("--in_tokens",
                    required=False,
                    type=int,
                    help="The sequence length for input tokens")
parser.add_argument("--out_tokens",
                    required=False,
                    type=int,
                    help="The sequence length for output tokens")
parser.add_argument("--count",
                    required=False,
                    type=int,
                    help="Number of requests sent")
parser.add_argument("--profile",
                    required=False,
                    type=str,
                    help="Path to profile json for pre-configured tests")
args = parser.parse_args()


def flatten_object_on_keys(obj, flatten_keys):
    """ Converts complex object into array of objects with individual parameter values in the flatten_keys """
    keys = obj.keys()
    values = []
    for key, val in obj.items():
        if isinstance(val, (list, tuple, set)) and key in flatten_keys:
            values.append(val)
        else:
            values.append([val])
    output = []
    for combination in product(*values):
        output.append(dict(zip(keys, combination)))
    return output


class LMITestSetupBuilder:
    """ Builds the test setup configuration for the test runner to run """

    def __init__(self, namespace):
        """
        :param namespace: Namespace object representing the parser arguments
        """
        self.setup_iterable_opts = [
            "engine", "model", "dtype", "tensor_parallel"
        ]
        self.config = {}
        self.config_from_args(namespace)
        if namespace.profile is not None:
            self.config_from_profile(namespace.profile)
        self._validate_config()

    def config_from_profile(self, profile_path):
        file = open(profile_path)
        profile_config = json.load(file)
        for key, value in profile_config.items():
            if key not in self.config or self.config[key] is None:
                """ profile parameters will not overwrite cli arguments """
                self.config[key] = value
        file.close()
        del self.config["profile"]

    def config_from_args(self, namespace):
        self.config = namespace.__dict__

    def get_test_series(self):
        return flatten_object_on_keys(self.config, self.setup_iterable_opts)

    def _validate_config(self):
        required_parameters = ["test_series", "docker_image", "model"]
        extended_requirements_test_series = ["performance"]
        for param in required_parameters:
            if param not in self.config or self.config[param] is None:
                raise AttributeError(
                    f"The following parameters must be set in config "
                    f"with profile or args:{required_parameters}")
        if self.config["test_series"] in extended_requirements_test_series:
            self._validate_test_series_config(self.config["test_series"])

    def _validate_test_series_config(self, series):
        required_parameters = []
        if series == "performance":
            required_parameters = [
                "engine", "dtype", "tensor_parallel", "batch_size",
                "out_tokens", "count"
            ]
        for param in required_parameters:
            if param not in self.config or self.config[param] is None:
                raise AttributeError(
                    f"The following extended parameters must be set in config for the {series} tests "
                    f"with profile or args:{required_parameters}")


class LMITestRunner:
    """ Runs a series of tests based on a LMITestSetupBuilder test series """

    def __init__(self, series):
        """
        :param series: LMITestSetupBuilder test series object
        """
        self.test_series = series
        self.cpu_mem_pid = None
        self.errors = []
        self.test_iterable_opts = [
            "in_tokens", "out_tokens", "count", "batch_size"
        ]
        self.setup_keys = ["engine", "dtype", "tensor_parallel"]
        self.test_sequence_keys = self.setup_keys + [
            "batch_size", "in_tokens", "out_tokens", "count", "cpu_memory"
        ]

    def arg_builder(self, obj):
        positional_args = ["test_series", "model"]
        output = []
        for arg in positional_args:
            if arg not in obj.keys():
                raise AttributeError(
                    f"LMITestRunner requires the following arguments: {positional_args}"
                )
            output.append(obj[arg])
        return output

    def kwarg_builder(self, obj, keys):
        output = []
        for kwarg in keys:
            if kwarg in obj.keys() and obj[kwarg] is not None:
                output.append(f"--{kwarg}")
                output.append(obj[kwarg])
        return output

    def get_script_args(self, obj, keys):
        args = self.arg_builder(obj)
        kwargs = self.kwarg_builder(obj, keys)
        combined = args + kwargs
        arg_string = " ".join(str(arg) for arg in combined)
        return arg_string

    def prepare_settings(self, test):
        args = self.get_script_args(test, self.setup_keys)
        command = f"python3 llm/prepare.py {args}"
        logging.info(command)
        sp.call(command, shell=True)

    def pull_docker_image(self, test):
        command = f"docker pull {test['docker_image']}"
        sp.call(command, shell=True)

    def launch_container(self, test):
        command = f"./launch_container.sh {test['docker_image']} {os.getcwd()}/models {test['platform']} " \
                  f"serve -m test=file:/opt/ml/model/test/"
        logging.info(command)
        try:
            output = sp.check_output([command], shell=True, stderr=sp.STDOUT)
        except sp.CalledProcessError as cpe:
            self.errors.append(
                f"Error:LMITestRunner.launch_container: {cpe.output.decode()}")
        else:
            logging.info(output.decode())

    def teardown_test(self):
        self.teardown_cpu_monitor()
        self.clean_cpu_logs()
        self.teardown_container()

    def teardown_cpu_monitor(self):
        command = f"kill {self.cpu_mem_pid}"
        sp.call(command, shell=True)

    def teardown_container(self):
        command = "docker rm -f $(docker ps -aq)"
        sp.call([command], shell=True)

    def clean_cpu_logs(self):
        command = "rm llm/cpu.log"
        sp.call(command, shell=True)

    def clean_metrics(self):
        command = "rm llm/metrics.log"
        sp.call(command, shell=True)

    def clean_test_setup(self):
        self.reset_error_logging()
        self.log_serving()
        self.teardown_test()

    def clean_test_sequence(self, sequence):
        self.reset_error_logging()
        logging.warning(f"Testing sequence failed while running: {sequence}")

    def reset_error_logging(self):
        for error in self.errors:
            self.log_errors(error)
        self.errors = []

    def check_errors(self):
        filename = "llm/errors.log"
        if os.path.exists(filename):
            command = f"rm {filename}"
            sp.call(command, shell=True)
            raise AssertionError(
                "Test Series failed with errors"
            )  # Raise error to fail test series in pipeline

    def log_errors(self, error):
        logging.warning(error)
        filename = "llm/errors.log"
        append_write = 'a' if os.path.exists(filename) else 'w'
        error_file = open(filename, append_write)
        error_file.write(error)
        error_file.close()

    def log_serving(self):
        command = "cat logs/*"
        sp.call(command, shell=True)

    def log_metrics(self, sequence):
        if not os.path.exists("llm/metrics.log"):
            logging.info(
                f"Metrics were not measured for this test sequence: {sequence}"
            )
            return
        command = "cat llm/metrics.log"
        if "log_metrics" in sequence:
            file = open("llm/metrics.log", "r")
            metrics = re.sub("'", r'"', file.readline())
            command = f'aws cloudwatch put-metric-data --namespace "LMIC_performance_{sequence["engine"]}" ' \
                      f'--region "us-east-1" --metric-data "{metrics}"'
        logging.info(command)
        sp.call(command, shell=True)
        self.clean_metrics()

    def set_cpu_monitor_pid(self):
        command = "ps aux | grep -Pm1 'cpu_memory_monitor' | awk -F ' ' '{print $2}'"
        self.cpu_mem_pid = int(sp.check_output(command, shell=True))

    def run_cpu_monitor(self):
        command = "nohup ./cpu_memory_monitor.sh > llm/cpu.log 2>&1 &"
        sp.call(command, shell=True)
        self.set_cpu_monitor_pid()

    def max_cpu_memory_used(self):
        total_memory = 0
        available_memory = 0
        with open("llm/cpu.log", "r") as cpu_log:
            for index, line in enumerate(cpu_log):
                if index == 0:  # First row of cpu mem logs contains the total cpu memory value
                    total_memory = int(line.split()[1])
                    available_memory = total_memory
                else:
                    available_memory = min(int(line.split()[1]),
                                           available_memory)
        if total_memory == 0:
            return total_memory
        return total_memory - available_memory

    def run_tests(self):
        for test in self.test_series:
            self.run_cpu_monitor()
            self.prepare_settings(test)
            self.pull_docker_image(test)
            self.launch_container(test)
            if len(self.errors) > 0:
                self.clean_test_setup()
                continue
            self.run_test_sequences(test)
            self.teardown_test()
        self.check_errors()

    def run_test_sequences(self, test_setup):
        test_sequences = flatten_object_on_keys(test_setup,
                                                self.test_iterable_opts)
        for test_sequence in test_sequences:
            test_sequence["cpu_memory"] = self.max_cpu_memory_used()
            self.run_client_requests(test_sequence)
            if len(self.errors) > 0:
                self.clean_test_sequence(test_sequence)
                continue
            self.log_metrics(test_sequence)

    def run_client_requests(self, test):
        test["cpu_memory"] = self.max_cpu_memory_used()
        positional_args = self.get_script_args(test, self.test_sequence_keys)
        command = f"python3 llm/client.py {positional_args}"
        try:
            output = sp.check_output(command, shell=True, stderr=sp.STDOUT)
        except sp.CalledProcessError as cpe:
            self.errors.append(
                f"Error:LMITestRunner.run_client_requests: {cpe.output.decode()}"
            )
        else:
            logging.info(output.decode())


if __name__ == "__main__":
    test_series = LMITestSetupBuilder(args).get_test_series()
    test_runner = LMITestRunner(test_series)
    test_runner.run_tests()
