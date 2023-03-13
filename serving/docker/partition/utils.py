import os
import json
import glob
import zipfile

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29761

FILES_TO_EXTRACT = [
    'djl_python/', 'djl_python/__init__.py', 'djl_python/deepspeed.py',
    'djl_python/inputs.py', 'djl_python/outputs.py', 'djl_python/pair_list.py',
    'djl_python/np_util.py', 'djl_python/service_loader.py',
    'djl_python/fastertransformer.py'
]


def get_python_executable():
    python_executable = os.environ.get("PYTHON_EXECUTABLE")
    if python_executable is None:
        python_executable = "python3"

    return python_executable


def get_partition_cmd(is_mpi_mode, properties):
    if is_mpi_mode:
        return [
            "mpirun", "-N",
            properties.get("tensor_parallel_degree", 1), "--allow-run-as-root",
            "--mca", "btl_vader_single_copy_mechanism", "none", "--tag-output",
            "-x", "FI_PROVIDER=efa", "-x", "RDMAV_FORK_SAFE=1", "-x",
            "FI_EFA_USE_DEVICE_RDMA=1", "-x", "LD_LIBRARY_PATH", "-x",
            f"MASTER_ADDR={MASTER_ADDR}", "-x", f"MASTER_PORT={MASTER_PORT}",
            "-x", "PYTHONPATH",
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
    engine = properties['engine']
    if engine == 'DeepSpeed':
        checkpoint_path = properties.get('save_mp_checkpoint_path')
        checkpoint_json = os.path.join(checkpoint_path,
                                       'ds_inference_config.json')
        if not os.path.exists(checkpoint_json):
            raise Exception('Partition was not successful')

        return {
            'option.model_dir': checkpoint_path,
            'option.checkpoint': 'ds_inference_config.json',
        }
    else:
        return {}


def extract_python_jar(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    jar_files = glob.glob('/usr/local/djl-serving-*/lib/python-*.jar')

    with zipfile.ZipFile(jar_files[0], 'r') as zip:
        # Extracting only required files into a specific location.
        for file in FILES_TO_EXTRACT:
            zip.extract(file, path=target_dir)


def is_engine_mpi_mode(engine):
    if engine == 'DeepSpeed':
        return True
    elif engine == 'FasterTransformer':
        return False
    else:
        raise NotImplementedError(
            f'{engine} '
            f'engine is not supported for ahead of time partitioning.')
