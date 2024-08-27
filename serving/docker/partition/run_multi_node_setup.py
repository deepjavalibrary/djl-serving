#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import sys
import os
import logging
import argparse
import hashlib
import requests
import subprocess
import glob

from pathlib import Path

from properties_manager import PropertiesManager
from huggingface_hub import snapshot_download

from retrying import retry

from utils import extract_python_jar, get_download_dir, load_properties, get_djl_version_from_lib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiNodeSetupHandler:

    def __init__(self):
        leader_hostname = os.environ.get("DJL_LEADER_ADDR", "")
        if not leader_hostname:
            raise ValueError(
                f"Hostname for DJL_LEADER_ADDR unavailable. Cannot obtain model info. Multi-node setup will not proceed."
            )

        if os.environ.get("LWS_LEADER_ADDR"):
            # In LWS env, make sure leader address has suffix svc.cluster.local
            if not leader_hostname.endswith("svc.cluster.local"):
                leader_hostname = f"{leader_hostname.rstrip('.')}.svc.cluster.local"

        self.cluster_leader_hostname = leader_hostname
        self.cluster_management_port = 8888

        self.endpoint_url = f"http://{self.cluster_leader_hostname}:{self.cluster_management_port}/cluster/"

    def hash(self, model_id):
        try:
            sha256 = hashlib.sha256()
            sha256.update(model_id.encode("utf-8"))
            digest = sha256.digest()
            hex_digest = digest[:20].hex()
            return hex_digest
        except Exception as e:
            raise AssertionError(
                "Error occurred while hashing the input model_url: ", e)

    def get_cache_dir(self):
        cache_dir = os.environ.get("DJL_CACHE_DIR")
        if not cache_dir:
            home_dir = Path(os.path.expanduser("~"))
            if not os.access(home_dir, os.W_OK):
                cache_dir = Path(os.environ.get("TEMP", "/tmp"))
            else:
                cache_dir = home_dir / ".djl.ai"
        else:
            cache_dir = Path(cache_dir)

        return cache_dir

    @retry(stop_max_delay=60000, wait_fixed=2000)
    def get_model_info_and_download_model(self):
        """
        Queries the Leader Address in LWS to obtain model_name, and model_url.
        Downloads the model to cache
        """
        try:
            logger.info(
                f"Getting model info from leader node to download model")

            endpoint_url = self.endpoint_url + "models"
            response = requests.get(endpoint_url)

            if response.status_code == 200:
                data = response.json()
                if type(data) != dict:
                    raise ValueError(
                        f"Expected a json dictionary response but received data of type: {type(data)}"
                    )

                for model_id, model_url in data.items():
                    logger.info(f"Model Name: {model_id}")
                    logger.info(f"Model URL: {model_url}")

                    if not model_url.startswith("s3://"):
                        # A path to serving.properties is returned
                        properties = load_properties(model_url.lstrip("file:"))
                        logger.info(f"properties: {properties}")
                        if "option.model_id" in properties:
                            model_url = properties["option.model_id"]
                        elif "OPTION_MODEL_ID" in os.environ:
                            model_url = os.environ.get("OPTION_MODEL_ID")
                    self.download_model_from_s3(model_id, model_url)
                    return True
            else:
                logger.error(
                    "Failed to fetch data from the endpoint. Retrying.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def download_model_from_s3(self, model_id, model_url):
        """
        Downloads model from s3 given s3 URL using s5cmd
        """
        logger.info("Downloading model...")

        if not model_url or not model_url.startswith("s3://"):
            logger.warn(
                f"Model {model_id} with URL {model_url} to be downloaded is not s3. Will attempt to download from HF hub."
            )
            return

        download_dir = os.environ.get("SERVING_DOWNLOAD_DIR", "")

        if not download_dir:
            download_dir = Path(
                self.get_cache_dir()) / "download" / self.hash(model_url)
            download_dir.mkdir(parents=True, exist_ok=True)

        if Path("/opt/djl/bin/s5cmd").is_file():
            if not model_url.endswith("*"):
                if model_url.endswith("/"):
                    model_url = model_url + "*"
                else:
                    model_url = model_url + "/*"

            commands = [
                "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync", model_url,
                download_dir
            ]
        else:
            commands = ["aws", "s3", "sync", model_url, download_dir]

        subprocess.run(commands)

        if not glob.glob(os.path.join(download_dir, "*")):
            raise Exception("Model download from s3url failed")

        logging.info(f"Model download complete.")
        return True

    @retry(stop_max_delay=60000, wait_fixed=2000)
    def get_and_write_cluster_ssh_key(self):
        """
        Gets ssh key generated by the leader node, and writes it to authorized keys.
        Private key may be necessary for bi-directional communication back to leader node, when using mpirun
        """
        logger.info(f"Querying ssh key from leader node")

        endpoint_url = self.endpoint_url + "sshpublickey"

        try:
            response = requests.get(endpoint_url)

            if response.status_code == 200:
                file_data = response.content

                home_dir = Path.home()
                ssh_dir = home_dir / ".ssh"
                ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

                authorized_keys_path = ssh_dir / "authorized_keys"

                with open(authorized_keys_path, "wb") as file:
                    file.write(file_data)

                authorized_keys_path.chmod(0o600)
            else:
                logger.info("Failed to fetch data from the endpoint.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    @retry(stop_max_delay=60000, wait_fixed=2000)
    def report_status_to_leader_node(self):
        """
        Reports status 'OK' back to leader node when setup is complete on worker node
        """
        logger.info(f"Reporting status to leader node")

        endpoint_url = self.endpoint_url + "status"
        params = {"message": "OK"}

        try:
            response = requests.get(endpoint_url, params=params)

            if response.status_code == 200:
                logger.info("Status reported successfully.")
                return True
            else:
                logger.error(
                    f"Failed to report status. Response code: {response.status_code}"
                )
        except Exception as e:
            logger.error(f"An error occurred: {e}")


def main():
    multi_node_setup_handler = MultiNodeSetupHandler()

    multi_node_setup_handler.get_model_info_and_download_model()

    extract_python_jar(
        target_dir=os.path.join(multi_node_setup_handler.get_cache_dir(),
                                "python", get_djl_version_from_lib()))
    multi_node_setup_handler.get_and_write_cluster_ssh_key()
    multi_node_setup_handler.report_status_to_leader_node()


if __name__ == "__main__":
    main()
