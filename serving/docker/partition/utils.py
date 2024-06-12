import os
import json
import glob
import zipfile
import tempfile
import logging

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29761

from transformers import AutoConfig, AutoTokenizer


def get_python_executable():
    python_executable = os.environ.get("PYTHON_EXECUTABLE")
    if python_executable is None:
        python_executable = "python3"

    return python_executable


def get_partition_cmd(is_mpi_mode, properties):
    if is_mpi_mode:
        return [
            "mpirun", "-N",
            properties.get("option.tensor_parallel_degree", "1"),
            "--allow-run-as-root", "--mca", "btl_vader_single_copy_mechanism",
            "none", "--tag-output", "-x", "FI_PROVIDER=efa", "-x",
            "RDMAV_FORK_SAFE=1", "-x", "FI_EFA_USE_DEVICE_RDMA=1", "-x",
            "LD_LIBRARY_PATH", "-x", f"MASTER_ADDR={MASTER_ADDR}", "-x",
            f"MASTER_PORT={MASTER_PORT}", "-x", "PYTHONPATH",
            get_python_executable(), "/opt/djl/partition/run_partition.py",
            "--properties",
            str(json.dumps(properties))
        ]
    else:
        return [
            get_python_executable(), "/opt/djl/partition/run_partition.py",
            "--properties",
            str(json.dumps(properties))
        ]


def get_engine_configs(properties):
    engine = properties.get('engine')
    configs = {'option.parallel_loading': True}
    if engine == 'DeepSpeed':
        configs['option.checkpoint'] = 'ds_inference_config.json'

    return configs


def extract_python_jar(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    jar_files = glob.glob('/usr/local/djl-serving-*/lib/python-*.jar')

    with zipfile.ZipFile(jar_files[0], 'r') as zip:
        zip.extractall(target_dir)


def is_engine_mpi_mode(engine):
    if engine == 'DeepSpeed':
        return True
    else:
        return False


def get_download_dir(properties_dir, suffix=""):
    tmp = tempfile.mkdtemp(suffix=suffix, prefix="download")
    download_dir = os.environ.get("SERVING_DOWNLOAD_DIR", tmp)
    if download_dir == "default":
        download_dir = properties_dir

    return download_dir


def load_properties(properties_dir):
    properties = {}
    properties_file = os.path.join(properties_dir, 'serving.properties')
    if os.path.exists(properties_file):
        with open(properties_file, 'r') as f:
            for line in f:
                # ignoring line starting with #
                if line.startswith("#") or not line.strip():
                    continue
                key, value = line.strip().split('=', 1)
                key = key.strip()
                value = value.strip()
                properties[key] = value
    return properties


def update_kwargs_with_env_vars(kwargs):
    env_vars = os.environ
    for key, value in env_vars.items():
        if key.startswith("OPTION_"):
            key = key.lower()
            key = "option." + key[7:]
            if key == "option.entrypoint":
                key = "option.entryPoint"
            kwargs.setdefault(key, value)
    return kwargs


def read_hf_model_config(model_config_path: str, hf_configs):
    try:
        model_config = AutoConfig.from_pretrained(
            model_config_path,
            trust_remote_code=hf_configs.trust_remote_code,
            revision=hf_configs.revision)
        return model_config
    except Exception as e:
        logging.error(
            f"{model_config_path} does not contain a config.json or adapter_config.json for lora models. "
            f"This is required for loading huggingface models")
        raise e


def init_hf_tokenizer(model_id_or_path: str, hf_configs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        padding_size="left",
        trust_remote_code=hf_configs.trust_remote_code,
        revision=hf_configs.revision,
    )
    return tokenizer
