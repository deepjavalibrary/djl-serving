#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import subprocess
import logging
from random import uniform
from glob import glob
from time import sleep
import os


# This function could only be used in docker container
def download_s3(s3url, model_dir):
    logging.info(f"Start downloading {s3url} to {model_dir}...")
    try:
        proc_run = subprocess.run(
            ["/opt/djl/bin/./s5cmd", "cp", "--recursive", s3url, model_dir]
        )
        logging.info("Model downloading finished")
        proc_run.check_returncode()  # to throw the error in case there was one

    except subprocess.CalledProcessError as e:
        logging.error("Model download failed: Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr)
        raise  # FAIL FAST


def download_s3_with_file_lock(s3url, model_dir, timeout):
    sleep(round(uniform(0.00, 2.00), 2))
    if f"{model_dir}/DONE" not in glob(f"{model_dir}/*"):
        if f"{model_dir}/PROGRESS" not in glob(f"{model_dir}/*"):
            with open(f"{model_dir}/PROGRESS", "w") as f:
                f.write("download in progress")
            download_s3(s3url, model_dir)
            os.remove(f"{model_dir}/PROGRESS")
            with open(f"{model_dir}/DONE", "w") as f:
                f.write("download_complete")
        else:
            wait_time = 0
            while wait_time < timeout:
                if f"{model_dir}/PROGRESS" in glob(f"{model_dir}/*"):
                    sleep(1)
                    wait_time += 1
                else:
                    break
            if wait_time >= timeout:
                raise TimeoutError("Loading timeout!")


def download_s3_with_dist_barrier(s3url, model_dir, rank, timeout):
    import torch.distributed as dist
    if rank == 0:
        download_s3_with_file_lock(s3url, model_dir, timeout)
    dist.barrier()
